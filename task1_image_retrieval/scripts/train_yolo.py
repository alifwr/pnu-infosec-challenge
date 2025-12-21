import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

import sys
from pathlib import Path

# Fix import path
sys.path.append(str(Path(__file__).parent.parent))

from utils.detection_loss import DetectionLoss, make_anchors
from utils.metrics import mean_average_precision

from modules.detection_dataset import YOLODataset, yolo_collate_fn
from models.loaders import load_model

LEARNING_RATE = 2e-4
BATCH_SIZE = 16
IMAGE_SIZE = 416
NUM_EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
train_dir = "./dataset/indonesia_vehicle/train"
valid_dir = "./dataset/indonesia_vehicle/valid"
test_dir = "./dataset/indonesia_vehicle/test"

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
    collate_fn=yolo_collate_fn,  # CRITICAL for object detection
    pin_memory=True,
)
valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=BATCH_SIZE,
    num_workers=2,
    shuffle=True,
    collate_fn=yolo_collate_fn,  # CRITICAL for object detection
    pin_memory=True,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=2,
    shuffle=False,
    collate_fn=yolo_collate_fn,  # CRITICAL for object detection
    pin_memory=True,
)

model = load_model("yolo_11", "configs/yolo_11.json", "weights/yolo11m_weights.pth").to(
    DEVICE
)

# # Freeze Backbone (b0 - b10)
# print("Freezing backbone layers...")
# for i in range(11):
#     layer_name = f"b{i}"
#     if hasattr(model, layer_name):
#         layer = getattr(model, layer_name)
#         for param in layer.parameters():
#             param.requires_grad = False
# print("Backbone layers frozen.")

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = DetectionLoss(nc=1)

print(f"Starting training on {DEVICE}...")

history = {"train_loss": [], "val_loss": [], "epoch": []}


for epoch in range(NUM_EPOCHS):
    model.train()
    mean_loss = []

    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", leave=True)

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(DEVICE)

        out = model(x)
        loss = loss_fn(out, y)

        optimizer.zero_grad()
        loss.backward()
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
            out = model(x)
            loss = loss_fn(out, y)
            val_loss.append(loss.item())
            val_loop.set_postfix(val_loss=loss.item())

    avg_val_loss = sum(val_loss) / len(val_loss)
    print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}")
    history["val_loss"].append(avg_val_loss)


torch.save(model.state_dict(), "yolo_custom.pth")
print("Model saved!")

print("Starting testing...")
model.eval()
test_loss = []
test_loop = tqdm(test_loader, desc="Testing", leave=True)

with torch.no_grad():
    for x, y in test_loop:
        x = x.to(DEVICE)
        # y processed in loss_fn
        out = model(x)
        loss = loss_fn(out, y)
        test_loss.append(loss.item())
        test_loop.set_postfix(test_loss=loss.item())

print(f"Final Test Loss: {sum(test_loss) / len(test_loss):.4f}")

test_idx = 0
all_test_pred_boxes = []
all_test_true_boxes = []

print("Calculating Test mAP...")
with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Testing mAP", leave=True):
        x = x.to(DEVICE)
        out = model(x)

        anchors, strides = make_anchors(out, loss_fn.strides, 0.5)
        preds_cat = []
        for p in out:
            preds_cat.append(p.flatten(2).permute(0, 2, 1))
        preds_cat = torch.cat(preds_cat, 1)

        box_channels = 4 * 16
        pred_box_dist = preds_cat[:, :, :box_channels]
        pred_cls = preds_cat[:, :, box_channels:]

        b, n, _ = pred_box_dist.shape
        pred_box_dist = pred_box_dist.view(b, n, 4, 16).softmax(3)
        proj = torch.arange(16, dtype=torch.float, device=DEVICE)
        pred_box_val = pred_box_dist.matmul(proj)

        lt, rb = torch.chunk(pred_box_val, 2, 2)
        x1y1 = anchors.unsqueeze(0) - lt
        x2y2 = anchors.unsqueeze(0) + rb
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1

        c_xy = c_xy * strides.unsqueeze(0)
        wh = wh * strides.unsqueeze(0)
        decoded_boxes = torch.cat((c_xy, wh), 2)

        pred_cls = pred_cls.sigmoid()

        for i in range(b):
            conf, cls_id = pred_cls[i].max(1)
            mask = conf > 0.1

            boxes = decoded_boxes[i][mask]
            scores = conf[mask]
            classes = cls_id[mask]

            b_x = boxes[:, 0]
            b_y = boxes[:, 1]
            b_w = boxes[:, 2]
            b_h = boxes[:, 3]
            x1 = b_x - b_w / 2
            y1 = b_y - b_h / 2
            x2 = b_x + b_w / 2
            y2 = b_y + b_h / 2

            for j in range(len(boxes)):
                all_test_pred_boxes.append(
                    [
                        test_idx,
                        int(classes[j]),
                        float(scores[j]),
                        float(x1[j]),
                        float(y1[j]),
                        float(x2[j]),
                        float(y2[j]),
                    ]
                )

            if y[i].ndim == 2 and y[i].shape[0] > 0:
                gt = y[i]
                gt_x = gt[:, 1] * IMAGE_SIZE
                gt_y = gt[:, 2] * IMAGE_SIZE
                gt_w = gt[:, 3] * IMAGE_SIZE
                gt_h = gt[:, 4] * IMAGE_SIZE

                gt_x1 = gt_x - gt_w / 2
                gt_y1 = gt_y - gt_h / 2
                gt_x2 = gt_x + gt_w / 2
                gt_y2 = gt_y + gt_h / 2

                for k in range(len(gt)):
                    all_test_true_boxes.append(
                        [
                            test_idx,
                            int(gt[k, 0]),
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

print("Saving metrics and plots...")

csv_file = "training_metrics.csv"
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
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("training_metrics.png")
print("Plot saved to training_metrics.png")
