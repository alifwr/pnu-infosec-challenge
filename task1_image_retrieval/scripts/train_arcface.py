import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
import os
import json
import yaml
from pathlib import Path
import numpy as np
import sys

sys.path.append(str(Path(__file__).parent.parent))

# Import your dataset class
from utils.dataset import (
    FlatStructureTankDataset,
    get_train_transforms,
    get_val_transforms,
    analyze_dataset,
)
from utils.training_utilities import (
    train_epoch,
    evaluate,
    build_gallery,
    save_checkpoint,
)
from models.arcface import TankClassifier


def main():
    # Load config from YAML file
    config_path = os.path.join(
        str(Path(__file__).parent.parent), "configs", "arcface_config.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Add device (computed at runtime)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)

    # Save config
    with open(os.path.join(config["output_dir"], "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print("=" * 80)
    print("TANK CLASSIFIER TRAINING PIPELINE")
    print("=" * 80)
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k:20s}: {v}")

    # Set random seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Analyze dataset first
    print("\n" + "=" * 80)
    analyze_dataset(config["images_dir"])
    print("=" * 80)

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = FlatStructureTankDataset(
        images_dir=config["images_dir"],
        transform=get_train_transforms(config["image_size"]),
        mode="train",
        train_ratio=config["train_ratio"],
        gallery_ratio=config["gallery_ratio"],
        seed=config["seed"],
    )

    gallery_dataset = FlatStructureTankDataset(
        images_dir=config["images_dir"],
        transform=get_val_transforms(config["image_size"]),
        mode="gallery",
        train_ratio=config["train_ratio"],
        gallery_ratio=config["gallery_ratio"],
        seed=config["seed"],
    )

    test_dataset = FlatStructureTankDataset(
        images_dir=config["images_dir"],
        transform=get_val_transforms(config["image_size"]),
        mode="test",
        train_ratio=config["train_ratio"],
        gallery_ratio=config["gallery_ratio"],
        seed=config["seed"],
    )

    # Save label mappings
    label_info = {
        "tank_types": train_dataset.tank_types,
        "label_to_idx": train_dataset.label_to_idx,
        "idx_to_label": train_dataset.idx_to_label,
    }
    with open(os.path.join(config["output_dir"], "label_mapping.json"), "w") as f:
        json.dump(label_info, f, indent=4)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    # Create model
    print(f"\nCreating model with {len(train_dataset.tank_types)} classes...")
    model = TankClassifier(
        num_classes=len(train_dataset.tank_types),
        embedding_size=config["embedding_size"],
        backbone=config["backbone"],
        pretrained=True,
    )
    model = model.to(config["device"])

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"], eta_min=1e-6
    )

    # Mixed precision scaler
    scaler = GradScaler("cuda") if config["mixed_precision"] else None

    # Training loop
    best_acc = 0.0
    train_losses = []

    print("\n" + "=" * 80)
    print("TRAINING START")
    print("=" * 80)

    for epoch in range(config["num_epochs"]):
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, config["device"], epoch, scaler
        )
        train_losses.append(train_loss)

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{config['num_epochs']} - Loss: {train_loss:.4f}, LR: {current_lr:.6f}"
        )

        # Evaluate every 10 epochs or at the end
        if (epoch + 1) % 10 == 0 or (epoch + 1) == config["num_epochs"]:
            # Build gallery
            gallery_prototypes, gallery_embeddings = build_gallery(
                model, gallery_loader, config["device"], train_dataset.idx_to_label
            )

            # Evaluate
            test_acc, preds, labels = evaluate(
                model,
                test_loader,
                gallery_prototypes,
                config["device"],
                train_dataset.idx_to_label,
                train_dataset.label_to_idx,
            )

            print(f"\nTest Accuracy: {test_acc:.4f}")

            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_acc": best_acc,
                        "config": config,
                    },
                    config["output_dir"],
                    "best_model.pth",
                )

                # Save gallery prototypes
                torch.save(
                    {
                        "prototypes": gallery_prototypes,
                        "embeddings": gallery_embeddings,
                        "label_mapping": label_info,
                    },
                    os.path.join(config["output_dir"], "gallery.pth"),
                )
                print(f"New best model saved! Accuracy: {best_acc:.4f}")

        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": config,
                },
                config["output_dir"],
                f"checkpoint_epoch_{epoch + 1}.pth",
            )

    # Save final model
    save_checkpoint(
        {
            "epoch": config["num_epochs"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
            "config": config,
        },
        config["output_dir"],
        "final_model.pth",
    )

    # Save training history
    history = {"train_losses": train_losses, "best_acc": best_acc}
    with open(os.path.join(config["output_dir"], "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Test Accuracy: {best_acc:.4f}")
    print(f"Models saved to: {config['output_dir']}")


if __name__ == "__main__":
    main()
