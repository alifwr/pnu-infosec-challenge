import sys
import os
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from modules.detection_dataset import YOLODataset, yolo_collate_fn

# Constants
BATCH_SIZE = 4
IMAGE_SIZE = 640
TRAIN_DIR = "./dataset/custom_dataset/train"
OUTPUT_FILE = "dataset_visualization.png"


def denormalize_box(box, w, h):
    """Convert normalized xywh to absolute xyxy."""
    # box: [class, cx, cy, bw, bh]
    cx, cy, bw, bh = box[1], box[2], box[3], box[4]
    cx *= w
    cy *= h
    bw *= w
    bh *= h

    x1 = int(cx - bw / 2)
    y1 = int(cy - bh / 2)
    x2 = int(cx + bw / 2)
    y2 = int(cy + bh / 2)
    return x1, y1, x2, y2


def visualize():
    # Setup transform (resize to match training)
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )

    # Load dataset
    dataset = YOLODataset(
        img_dir=os.path.join(TRAIN_DIR, "images"),
        label_dir=os.path.join(TRAIN_DIR, "labels"),
        transform=transform,
    )

    if len(dataset) == 0:
        print(f"No images found in {TRAIN_DIR}")
        return

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=yolo_collate_fn
    )

    # Get one batch
    images, targets = next(iter(loader))

    # Prepare canvas
    # Convert tensor images back to numpy for cv2
    vis_images = []

    print(f"Visualizing batch of {len(images)} images...")

    for i in range(len(images)):
        img_tensor = images[i]
        target = targets[i]  # [N, 5]

        # Convert CxHxW -> HxWxC
        img_np = img_tensor.permute(1, 2, 0).numpy()
        # Scale to 0-255
        img_np = (img_np * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        h, w = img_np.shape[:2]

        # Draw boxes
        if len(target) > 0:
            for box in target:
                x1, y1, x2, y2 = denormalize_box(box, w, h)
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cls_id = int(box[0])
                cv2.putText(
                    img_np,
                    f"Class {cls_id}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        vis_images.append(img_np)

    # Concatenate images specific way (e.g. horizontal)
    final_img = np.hstack(vis_images)

    # Save
    cv2.imwrite(OUTPUT_FILE, final_img)
    print(f"Saved visualization to {OUTPUT_FILE}")


if __name__ == "__main__":
    visualize()
