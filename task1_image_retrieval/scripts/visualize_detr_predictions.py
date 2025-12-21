import torch
import cv2
import os
import sys
import random
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from models.loaders import load_model
from models.detector.rtdetr_v2.box_ops import box_cxcywh_to_xyxy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 640
CONF_THRESHOLD = 0.45
NUM_SAMPLES = 5
WEIGHTS_PATH = "weights/rtdetrv2_r34vd_120e_coco_ema.pth"
TEST_IMG_DIR = "./dataset/indonesia_vehicle/test/images"
OUTPUT_DIR = "./predictions_detr"


def load_trained_model():
    print("Initializing RT-DETR (nc=1)...")

    # Check if custom weights exist, otherwise warn (or could fallback to pretrained if desired)
    if not os.path.exists(WEIGHTS_PATH):
        print(
            f"Warning: Custom weights {WEIGHTS_PATH} not found. Using pretrained weights if available in config or None."
        )
        weights = "weights/rtdetrv2_r34vd_120e_coco_ema.pth"  # Fallback to base
    else:
        weights = WEIGHTS_PATH
        print(f"Loading weights from {weights}")

    model = load_model("rtdetr_v2", "configs/rtdetr_v2.json", weight_path=weights)
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
    pred_logits = out["pred_logits"]  # [B, N, num_classes]
    pred_boxes = out["pred_boxes"]  # [B, N, 4]

    prob = pred_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(
        prob.view(pred_logits.shape[0], -1), 100, dim=1
    )
    scores = topk_values[0]  # Batch 0
    topk_boxes = topk_indexes[0] // prob.shape[2]
    labels = topk_indexes[0] % prob.shape[2]

    # Gather boxes
    boxes = torch.gather(pred_boxes[0], 0, topk_boxes.unsqueeze(-1).repeat(1, 4))

    # Convert cxcywh to xyxy
    boxes = box_cxcywh_to_xyxy(boxes)

    # DETR boxes are normalized [0, 1]. Scale to Image Size (640) used during inference first?
    # Actually RT-DETR usually operates on normalized boxes.
    # But for visualization we need to map to Original Image.

    # First, separate x,y
    boxes_np = boxes.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    orig_h, orig_w = original_shape
    results = []

    for i in range(len(scores_np)):
        score = scores_np[i]
        if score < CONF_THRESHOLD:
            continue

        label = labels_np[i]
        box = boxes_np[i]  # x1, y1, x2, y2 normalized

        # Scale to original image
        x1 = int(box[0] * orig_w)
        y1 = int(box[1] * orig_h)
        x2 = int(box[2] * orig_w)
        y2 = int(box[3] * orig_h)

        results.append(
            {"bbox": [x1, y1, x2, y2], "score": float(score), "class": int(label)}
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
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for DETR

        # Label
        label = f"Cls {cls} {score:.2f}"
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img_bgr, (x1, y1 - 20), (x1 + t_size[0], y1), (0, 0, 255), -1)
        cv2.putText(
            img_bgr,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
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
        save_path = os.path.join(OUTPUT_DIR, f"detr_pred_{fname}")
        draw_predictions(orig_img, preds, save_path)

    print("Done!")


if __name__ == "__main__":
    main()
