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
import pandas as pd
import time
import wandb

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
    validate_loss,
    evaluate,
    build_gallery,
    save_checkpoint,
)
from models.arcface import VehicleClassifier


def train_model(backbone_name, base_config):
    """
    Trains and benchmarks a single model defined by backbone_name.
    Returns a dictionary of benchmark results.
    """
    # Create a copy of config to avoid side effects
    config = base_config.copy()
    config["backbone"] = backbone_name

    # Update output directory specific to this backbone
    config["output_dir"] = os.path.join(base_config["output_dir"], backbone_name)
    os.makedirs(config["output_dir"], exist_ok=True)

    # Save config
    with open(os.path.join(config["output_dir"], "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Initialize wandb
    try:
        wandb.init(
            project="vehicle_classification",
            name=f"arcface_{config['backbone']}",
            config=config,
            reinit=True,
        )
        print(f"Logged into WandB successfully for {backbone_name}.")
    except Exception as e:
        print(f"WandB init failed: {e}")
        print("Continuing without WandB logging...")

    print("=" * 80)
    print(f"TRAINING MODEL: {backbone_name}")
    print("=" * 80)

    # Set random seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

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
        "vehicle_types": train_dataset.vehicle_types,
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
    print(f"\nCreating model with {len(train_dataset.vehicle_types)} classes...")
    model = VehicleClassifier(
        num_classes=len(train_dataset.vehicle_types),
        embedding_size=config["embedding_size"],
        backbone=config["backbone"],
        pretrained=True,
    )
    model = model.to(config["device"])

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

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

    print("\n" + "-" * 60)
    print(f"Start Training for {backbone_name}")
    print("-" * 60)

    for epoch in range(config["num_epochs"]):
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, config["device"], epoch, scaler
        )
        train_losses.append(train_loss)

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Validation Loss
        val_loss = validate_loss(model, gallery_loader, criterion, config["device"])

        print(
            f"Epoch {epoch + 1}/{config['num_epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}"
        )

        # Log training metrics
        if wandb.run:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                    "epoch": epoch + 1,
                }
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

            # Log validation metrics
            if wandb.run:
                wandb.log({"test_accuracy": test_acc, "epoch": epoch + 1})

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

    # Save training history
    history = {"train_losses": train_losses, "best_acc": best_acc}
    with open(os.path.join(config["output_dir"], "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # ---------------------------------------------------------
    # Benchmarking
    # ---------------------------------------------------------
    print("\nRunning Benchmark...")

    # Load best model for benchmarking
    best_model_path = os.path.join(config["output_dir"], "best_model.pth")
    if os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            print("Loaded best model for benchmarking.")
        except RuntimeError as e:
            print(f"Warning: Could not load best_model.pth: {e}")

    model.eval()

    # Measure inference time
    total_images = 0
    with torch.no_grad():
        # Warmup
        for i, (images, _) in enumerate(test_loader):
            if i >= 5:
                break
            images = images.to(config["device"])
            _ = model(images)

        # Actual measurement
        start_time = time.time()
        for images, _ in test_loader:
            images = images.to(config["device"])
            _ = model(images)
            total_images += images.size(0)
        end_time = time.time()

    inference_time_ms = (
        ((end_time - start_time) / total_images) * 1000 if total_images > 0 else 0.0
    )

    # Model parameters
    total_params_million = sum(p.numel() for p in model.parameters()) / 1e6

    # Close wandb run for this model
    if wandb.run:
        wandb.finish()

    return {
        "Model": backbone_name,
        "Accuracy": best_acc,
        "Inference Time (ms)": round(inference_time_ms, 2),
        "Parameters (M)": round(total_params_million, 2),
    }


def main():
    # Load base config
    config_path = os.path.join(
        str(Path(__file__).parent.parent), "configs", "arcface_config.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Add device
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Analyze dataset once
    print("\n" + "=" * 80)
    analyze_dataset(config["images_dir"])
    print("=" * 80)

    # Define backbones to train
    backbones = ["efficientnet_b3", "swin_tiny_patch4_window7_224"]

    all_results = []

    for backbone in backbones:
        try:
            result = train_model(backbone, config)
            all_results.append(result)
        except Exception as e:
            print(f"\nERROR: Training failed for {backbone}: {e}")
            import traceback

            traceback.print_exc()

    # ---------------------------------------------------------
    # Final Consolidated Report
    # ---------------------------------------------------------
    if all_results:
        df = pd.DataFrame(all_results)
        output_csv_path = "benchmark_report_classification.csv"
        df.to_csv(output_csv_path, index=False)

        print("\n" + "=" * 50)
        print("FINAL BENCHMARK REPORT")
        print("=" * 50)
        print(df.to_string(index=False))
        print(f"\nReport saved to {os.path.abspath(output_csv_path)}")
    else:
        print("\nNo results to report.")


if __name__ == "__main__":
    main()
