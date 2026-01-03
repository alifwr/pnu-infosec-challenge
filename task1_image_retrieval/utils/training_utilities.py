import torch
import os
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp.autocast_mode import autocast


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, checkpoint_dir, filename="checkpoint.pth"):
    """Save model checkpoint"""
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint.get("epoch", 0), checkpoint.get("best_acc", 0.0)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler=None):
    """Train for one epoch"""
    model.train()

    losses = AverageMeter()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [TRAIN]")

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Mixed precision training
        if scaler is not None:
            with autocast("cuda"):
                logits, embeddings = model(images, labels)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, embeddings = model(images, labels)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))

        # Update progress bar
        pbar.set_postfix({"loss": f"{losses.avg:.4f}"})

    return losses.avg


def build_gallery(model, gallery_loader, device, idx_to_label):
    """Build gallery prototypes from gallery set"""
    model.eval()

    gallery_embeddings = {label: [] for label in idx_to_label.values()}

    print("\nBuilding gallery...")
    with torch.no_grad():
        for images, labels in tqdm(gallery_loader, desc="Gallery"):
            images = images.to(device)

            # Extract embeddings
            embeddings = model(images)  # Returns normalized embeddings

            # Store by class
            for emb, label in zip(embeddings, labels):
                label_name = idx_to_label[label.item()]
                gallery_embeddings[label_name].append(emb.cpu())

    # Compute prototypes (average embeddings per class)
    gallery_prototypes = {}
    for label, embs in gallery_embeddings.items():
        if len(embs) > 0:
            embs = torch.stack(embs)
            # Normalize again after averaging
            gallery_prototypes[label] = F.normalize(embs.mean(dim=0), dim=0)
            print(f"  {label}: {len(embs)} samples")

    return gallery_prototypes, gallery_embeddings


def evaluate(
    model, test_loader, gallery_prototypes, device, idx_to_label, label_to_idx
):
    """Evaluate model on test set using gallery"""
    model.eval()

    correct = 0
    total = 0

    # For confusion matrix
    all_preds = []
    all_labels = []

    print("\nEvaluating on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test"):
            images = images.to(device)
            labels = labels.to(device)

            # Extract query embeddings
            query_embeddings = model(images)

            # Compare to gallery
            for query_emb, true_label in zip(query_embeddings, labels):
                # Compute similarities to all prototypes
                similarities = {}
                for tank_type, prototype in gallery_prototypes.items():
                    sim = F.cosine_similarity(
                        query_emb.unsqueeze(0), prototype.unsqueeze(0).to(device)
                    ).item()
                    similarities[tank_type] = sim

                # Get prediction
                predicted_type = max(similarities, key=lambda k: similarities[k])
                predicted_idx = label_to_idx[predicted_type]

                all_preds.append(predicted_idx)
                all_labels.append(true_label.item())

                if predicted_idx == true_label.item():
                    correct += 1
                total += 1

    accuracy = correct / total if total > 0 else 0.0

    # Print per-class accuracy
    print("\nPer-class accuracy:")
    class_correct = {label: 0 for label in idx_to_label.values()}
    class_total = {label: 0 for label in idx_to_label.values()}

    for pred, true in zip(all_preds, all_labels):
        label_name = idx_to_label[true]
        class_total[label_name] += 1
        if pred == true:
            class_correct[label_name] += 1

    for label in sorted(class_correct.keys()):
        if class_total[label] > 0:
            acc = class_correct[label] / class_total[label]
            print(
                f"  {label:30s}: {acc:.4f} ({class_correct[label]}/{class_total[label]})"
            )

    return accuracy, all_preds, all_labels
