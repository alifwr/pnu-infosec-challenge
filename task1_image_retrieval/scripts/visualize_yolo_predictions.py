import torch
import cv2
import os
import sys
import random
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from models.loaders import load_model
from utils.detection_loss import make_anchors

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 416
CONF_THRESHOLD = 0.25
NUM_SAMPLES = 5
WEIGHTS_PATH = "weights/yolo11m_weights.pth"
TEST_IMG_DIR = "./dataset/indonesia_vehicle/test/images"
OUTPUT_DIR = "./predictions"
STRIDES = [8, 16, 32]


def load_trained_model():
    print("Initializing YOLO11m (nc=1)...")
    # Using load_model as requested
    model = load_model("yolo_11", "configs/yolo_11.json", WEIGHTS_PATH)

    return model.to(DEVICE).eval()


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Resize
    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    # Normalize
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    return img_tensor.to(DEVICE), img, (h, w)


def decode_outputs(out, original_shape):
    # Verify strides
    # make_anchors expects strides as list
    anchors, strides_tensor = make_anchors(out, STRIDES, 0.5)

    preds_cat = []
    for p in out:
        preds_cat.append(p.flatten(2).permute(0, 2, 1))
    preds_cat = torch.cat(preds_cat, 1)

    # Split
    box_channels = 4 * 16
    pred_box_dist = preds_cat[:, :, :box_channels]
    pred_cls = preds_cat[:, :, box_channels:]

    # Box Distribution
    b, n, _ = pred_box_dist.shape
    pred_box_dist = pred_box_dist.view(b, n, 4, 16).softmax(3)
    proj = torch.arange(16, dtype=torch.float, device=DEVICE)
    pred_box_val = pred_box_dist.matmul(proj)

    # Decode to xywh relative to feature map
    lt, rb = torch.chunk(pred_box_val, 2, 2)
    x1y1 = anchors.unsqueeze(0) - lt
    x2y2 = anchors.unsqueeze(0) + rb
    c_xy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1

    # Scale up by stride
    c_xy = c_xy * strides_tensor.unsqueeze(0)
    wh = wh * strides_tensor.unsqueeze(0)

    # Concatenate xywh
    decoded_boxes = torch.cat((c_xy, wh), 2)  # (B, N, 4)

    # Cls
    pred_cls = pred_cls.sigmoid()

    # Process batch 0 only
    conf, cls_id = pred_cls[0].max(1)
    mask = conf > CONF_THRESHOLD

    boxes = decoded_boxes[0][mask]
    scores = conf[mask]
    classes = cls_id[mask]

    if len(boxes) == 0:
        return []

    # Convert from 416x416 to Original Image Size
    orig_h, orig_w = original_shape
    scale_x = orig_w / IMAGE_SIZE
    scale_y = orig_h / IMAGE_SIZE

    results = []
    for i in range(len(boxes)):
        bx, by, bw, bh = boxes[i].tolist()

        # Scale back
        bx *= scale_x
        by *= scale_y
        bw *= scale_x
        bh *= scale_y

        x1 = int(bx - bw / 2)
        y1 = int(by - bh / 2)
        x2 = int(bx + bw / 2)
        y2 = int(by + bh / 2)

        results.append(
            {
                "bbox": [x1, y1, x2, y2],
                "score": float(scores[i]),
                "class": int(classes[i]),
            }
        )

    return results


def draw_predictions(img, predictions, output_path):
    # img is RGB
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for pred in predictions:
        x1, y1, x2, y2 = pred["bbox"]
        score = pred["score"]
        cls = pred["class"]

        # Draw box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label
        label = f"Cls {cls} {score:.2f}"
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img_bgr, (x1, y1 - 20), (x1 + t_size[0], y1), (0, 255, 0), -1)
        cv2.putText(
            img_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )

    cv2.imwrite(output_path, img_bgr)
    print(f"Saved {output_path}")


def main():
    if not os.path.exists(TEST_IMG_DIR):
        print(f"Test directory {TEST_IMG_DIR} not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_trained_model()

    all_imgs = [
        os.path.join(TEST_IMG_DIR, f)
        for f in os.listdir(TEST_IMG_DIR)
        if f.endswith((".jpg", ".png"))
    ]
    if len(all_imgs) == 0:
        print("No images found in test directory.")
        return

    selected_imgs = random.sample(all_imgs, min(len(all_imgs), NUM_SAMPLES))

    print(f"Visualizing {len(selected_imgs)} sample images...")

    for i, img_path in enumerate(selected_imgs):
        inp, orig_img, orig_shape = preprocess_image(img_path)
        if inp is None:
            continue

        with torch.no_grad():
            out = model(inp)

        preds = decode_outputs(out, orig_shape)

        fname = os.path.basename(img_path)
        save_path = os.path.join(OUTPUT_DIR, f"pred_{fname}")
        draw_predictions(orig_img, preds, save_path)

    print("Done!")


if __name__ == "__main__":
    main()
