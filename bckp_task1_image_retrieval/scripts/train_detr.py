import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import os

import sys
from pathlib import Path

# Fix import path
sys.path.append(str(Path(__file__).parent.parent))

from models.detector.rtdetr_v2.box_ops import box_cxcywh_to_xyxy
from models.loaders import load_model
from utils.detr_loss import HungarianMatcher, SetCriterion
from utils.metrics import mean_average_precision
from utils.early_stopping import EarlyStopping

from modules.detection_dataset import YOLODataset, yolo_collate_fn

LEARNING_RATE = 1e-4  # DETR often needs lower LR
BATCH_SIZE = 16  # RT-DETR can be heavy
IMAGE_SIZE = 640  # RT-DETR usually 640
NUM_EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
train_dir = "./dataset/cctv_dataset/train"
valid_dir = "./dataset/cctv_dataset/valid"
test_dir = "./dataset/cctv_dataset/test"

transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)

train_dataset = YOLODataset(
    img_dir=train_dir + "/images", label_dir=train_dir + "/labels", transform=transform
)
valid_dataset = YOLODataset(
    img_dir=valid_dir + "/images", label_dir=valid_dir + "/labels", transform=transform
)
test_dataset = YOLODataset(
    img_dir=test_dir + "/images", label_dir=test_dir + "/labels", transform=transform
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=2,
    shuffle=True,
    collate_fn=yolo_collate_fn,
    pin_memory=True,
)
valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=BATCH_SIZE,
    num_workers=2,
    shuffle=True,
    collate_fn=yolo_collate_fn,
    pin_memory=True,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=2,
    shuffle=False,
    collate_fn=yolo_collate_fn,
    pin_memory=True,
)

weights_path = "weights/rtdetrv2_r34vd_120e_coco_ema.pth"

model = load_model(
    "rtdetr_v2",
    config_path="configs/rtdetr_v2.json",
    weight_path=weights_path,
).to(DEVICE)

# Freeze backbone and encoder
print("Freezing backbone and encoder...")
for param in model.backbone.parameters():
    param.requires_grad = False
for param in model.encoder.parameters():
    param.requires_grad = False

print("Backbone and Encoder frozen.")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")

matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
weight_dict = {"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2}
losses = ["labels", "boxes", "cardinality"]
criterion = SetCriterion(
    1, matcher, weight_dict, eos_coef=0.1, losses=["labels", "boxes"]
).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Initialize Early Stopping
early_stopping = EarlyStopping(patience=10, verbose=True, path="rtdetr_custom.pth")

print(f"Starting training on {DEVICE}...")

history = {"train_loss": [], "val_loss": [], "epoch": []}


def format_targets(batch_targets, device):
    new_targets = []
    for t in batch_targets:
        t = t.to(device)
        if len(t) > 0:
            labels = torch.zeros_like(t[:, 0], dtype=torch.long)
            boxes = t[:, 1:]
            new_targets.append({"labels": labels, "boxes": boxes})
        else:
            new_targets.append(
                {
                    "labels": torch.tensor([], dtype=torch.long, device=device),
                    "boxes": torch.tensor([], dtype=torch.float, device=device),
                }
            )
    return new_targets


for epoch in range(NUM_EPOCHS):
    model.train()
    mean_loss = []

    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", leave=True)

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(DEVICE)
        targets = format_targets(y, DEVICE)

        out = model(x, targets)

        loss_dict = criterion(out, targets)
        weight_dict = criterion.weight_dict
        loss = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        mean_loss.append(loss.item())
        loop.set_postfix(loss=loss.item())

    avg_train_loss = sum(mean_loss) / len(mean_loss)
    print(f"Epoch {epoch + 1} finished. Average Loss: {avg_train_loss:.4f}")
    history["train_loss"].append(avg_train_loss)
    history["epoch"].append(epoch + 1)

    # --- VALIDATION LOOP ---
    model.eval()
    val_loss = []
    val_loop = tqdm(
        valid_loader, desc=f"Validation {epoch + 1}/{NUM_EPOCHS}", leave=True
    )

    with torch.no_grad():
        for x, y in val_loop:
            x = x.to(DEVICE)
            targets = format_targets(y, DEVICE)

            out = model(x)
            loss_dict = criterion(out, targets)
            weight_dict = criterion.weight_dict
            loss = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            )

            val_loss.append(loss.item())
            val_loop.set_postfix(val_loss=loss.item())

    avg_val_loss = sum(val_loss) / len(val_loss)
    print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}")
    history["val_loss"].append(avg_val_loss)

    # Early Stopping check
    early_stopping(avg_val_loss, model)

    if early_stopping.early_stop:
        print("Early stopping")
        break

# Load the last checkpoint with the best model
model.load_state_dict(torch.load("rtdetr_custom.pth"))
print("Model saved!")

print("Starting testing...")
model.eval()

test_loss = []
test_loop = tqdm(test_loader, desc="Testing", leave=True)
with torch.no_grad():
    for x, y in test_loop:
        x = x.to(DEVICE)
        targets = format_targets(y, DEVICE)
        out = model(x)
        loss_dict = criterion(out, targets)
        weight_dict = criterion.weight_dict
        loss = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        test_loss.append(loss.item())
        test_loop.set_postfix(test_loss=loss.item())

print(f"Final Test Loss: {sum(test_loss) / len(test_loss):.4f}")

# Calculate mAP
print("Calculating Test mAP...")
all_test_pred_boxes = []
all_test_true_boxes = []
test_idx = 0

with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Testing mAP", leave=True):
        x = x.to(DEVICE)
        # Inference
        out = model(x)
        pred_logits = out["pred_logits"]
        pred_boxes = out["pred_boxes"]

        prob = pred_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(x.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // prob.shape[2]
        labels = topk_indexes % prob.shape[2]

        boxes = torch.gather(pred_boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        boxes = box_cxcywh_to_xyxy(boxes)
        boxes = boxes * IMAGE_SIZE

        batch_size = x.shape[0]
        for i in range(batch_size):
            b_scores = scores[i]
            b_labels = labels[i]
            b_boxes = boxes[i]

            mask = b_scores > 0.1
            b_scores = b_scores[mask]
            b_labels = b_labels[mask]
            b_boxes = b_boxes[mask]

            for j in range(len(b_scores)):
                all_test_pred_boxes.append(
                    [
                        test_idx,
                        int(b_labels[j]),
                        float(b_scores[j]),
                        float(b_boxes[j][0]),
                        float(b_boxes[j][1]),
                        float(b_boxes[j][2]),
                        float(b_boxes[j][3]),
                    ]
                )

            if y[i].ndim == 2 and y[i].shape[0] > 0:
                gt = y[i]
                gt_cls = gt[:, 0]
                gt_cx = gt[:, 1] * IMAGE_SIZE
                gt_cy = gt[:, 2] * IMAGE_SIZE
                gt_w = gt[:, 3] * IMAGE_SIZE
                gt_h = gt[:, 4] * IMAGE_SIZE

                gt_x1 = gt_cx - gt_w / 2
                gt_y1 = gt_cy - gt_h / 2
                gt_x2 = gt_cx + gt_w / 2
                gt_y2 = gt_cy + gt_h / 2

                for k in range(len(gt)):
                    all_test_true_boxes.append(
                        [
                            test_idx,
                            int(gt_cls[k]),
                            1.0,
                            float(gt_x1[k]),
                            float(gt_y1[k]),
                            float(gt_x2[k]),
                            float(gt_y2[k]),
                        ]
                    )
            test_idx += 1

test_map50 = mean_average_precision(
    all_test_pred_boxes,
    all_test_true_boxes,
    iou_threshold=0.5,
    box_format="corners",
    num_classes=1,
)
print(f"Final Test mAP@50: {test_map50:.4f}")

# Save Metrics
print("Saving metrics and plots...")
csv_file = "training_metrics_detr.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Validation Loss"])
    for i in range(len(history["epoch"])):
        writer.writerow(
            [history["epoch"][i], history["train_loss"][i], history["val_loss"][i]]
        )
print(f"Metrics saved to {csv_file}")

plt.figure(figsize=(10, 5))
plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
plt.plot(history["epoch"], history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("RT-DETR Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("training_metrics_detr.png")
print("Plot saved to training_metrics_detr.png")
