import argparse
import sys
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import custom modules
from models.arcface import VehicleClassifier  # noqa: E402


def get_transforms(image_size=224):
    """Validation/Inference transforms matching training"""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def load_classification_model(config_path, checkpoint_path, gallery_path, device):
    """Load the trained classification model and gallery"""
    print(f"Loading classification model from {checkpoint_path}...")

    # Load gallery to get metadata first
    if not os.path.exists(gallery_path):
        raise FileNotFoundError(f"Gallery file not found: {gallery_path}")

    gallery_data = torch.load(gallery_path, map_location=device)
    label_mapping = gallery_data["label_mapping"]
    idx_to_label = {int(k): v for k, v in label_mapping["idx_to_label"].items()}
    num_classes = len(idx_to_label)

    # Convert dictionary of prototypes to tensor
    gallery_dict = gallery_data["prototypes"]
    prototypes_list = []

    for i in range(num_classes):
        label = idx_to_label[i]
        if label in gallery_dict:
            prototypes_list.append(gallery_dict[label])
        else:
            print(f"Warning: No prototype found for class {label}")
            # Initialize random valid vector
            prototypes_list.append(F.normalize(torch.randn(512), dim=0))

    gallery_prototypes = torch.stack(prototypes_list).to(device)
    # Initialize model
    model = VehicleClassifier(
        num_classes=num_classes,
        embedding_size=512,  # Standard size used in training
        backbone="efficientnet_b3",  # We know this is the best model
        pretrained=False,  # No need to download weights, we load checkpoint
    )

    # Load weights
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, gallery_prototypes, idx_to_label


def classify_crop(model, crop, transform, device, gallery_prototypes):
    """Classify a single vehicle crop"""
    # Preprocess
    # Albumentations expects RGB numpy array
    transformed = transform(image=crop)["image"]
    input_tensor = transformed.unsqueeze(0).to(device)

    with torch.no_grad():
        # Get embedding (normalized)
        embedding = model(input_tensor)

        # Calculate similarity with gallery prototypes
        # embedding: [1, 512], prototypes: [num_classes, 512]
        # cosine similarity = dot product of normalized vectors
        similarity = F.linear(embedding, gallery_prototypes)

        # Get best match
        score, idx = torch.max(similarity, dim=1)

    return idx.item(), score.item()


def processed_video(
    video_path, output_path, detection_model_path, cls_model_dir, device="cuda"
):
    # 1. Setup paths
    cls_checkpoint = os.path.join(cls_model_dir, "best_model.pth")
    cls_gallery = os.path.join(cls_model_dir, "gallery.pth")

    # 2. Load Models
    print("Loading models...")
    # Object Detection
    det_model = YOLO(detection_model_path)

    # Image Classification
    cls_model, gallery_prototypes, idx_to_label = load_classification_model(
        None, cls_checkpoint, cls_gallery, device
    )
    transform = get_transforms()

    # 3. Open Video
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processing frame {frame_count}...", end="\r")

        # 4. Object Detection
        # Conf threshold 0.25 to be safe
        results = det_model(frame, conf=0.25, verbose=False)[0]

        # Prepare drawing
        annotated_frame = frame.copy()

        for box in results.boxes:
            # Get coords
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())  # 0 is vehicle for our model

            # Extract crop
            # Ensure within bounds
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 - x1 < 10 or y2 - y1 < 10:  # Skip tiny boxes
                continue

            crop = frame[y1:y2, x1:x2]
            # Convert BGR to RGB for classification
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            # 5. Classify
            class_idx, score = classify_crop(
                cls_model, crop_rgb, transform, device, gallery_prototypes
            )
            class_label = idx_to_label.get(class_idx, "Unknown")

            # 6. Draw
            # Color based on class hash
            color = (0, 255, 0)

            # Draw Box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Draw Label Background
            label_text = f"{class_label} ({score:.2f})"
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                annotated_frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1
            )

            # Draw Label Text
            cv2.putText(
                annotated_frame,
                label_text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        out.write(annotated_frame)

    cap.release()
    out.release()
    print("\nProcessing complete!")
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Final Experiment: Detection + Identity Classification"
    )
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument(
        "--output", type=str, default="final_output.mp4", help="Path to output video"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    # Hardcoded paths based on project structure
    DETECTION_MODEL = str(
        PROJECT_ROOT / "vehicle_detection/vehicles_yolo10l/weights/best.pt"
    )
    CLASSIFICATION_DIR = str(
        PROJECT_ROOT / "outputs/indonesian_vehicles/efficientnet_b3"
    )

    if not os.path.exists(DETECTION_MODEL):
        print(f"Error: Detection model not found at {DETECTION_MODEL}")
        sys.exit(1)

    if not os.path.exists(CLASSIFICATION_DIR):
        print(f"Error: Classification model dir not found at {CLASSIFICATION_DIR}")
        sys.exit(1)

    processed_video(
        args.video, args.output, DETECTION_MODEL, CLASSIFICATION_DIR, args.device
    )
