import os
import cv2
import torch
import yaml
import glob
import argparse
import random
from pathlib import Path
from groundingdino.util.inference import load_model, load_image, predict


def main():
    # Load default config or from args
    config_path = "configs/video_dataset.yaml"
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Setup paths
    videos_dir = config.get("videos_dir", "data/videos")
    output_dir = config.get("output_dir", "dataset/yolo_vehicles")
    grounding_config = config.get(
        "grounding_config", "configs/GroundingDINO_SwinT_OGC.py"
    )
    grounding_weights = config.get(
        "grounding_weights", "weights/groundingdino_swint_ogc.pth"
    )

    # Check weights
    if not os.path.exists(grounding_weights):
        print(f"Weights not found at {grounding_weights}")
        return

    # Create directories
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    # Load model
    print("Loading GroundingDINO model...")
    model = load_model(grounding_config, grounding_weights)

    # Get video files
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(videos_dir, ext)))

    print(f"Found {len(video_files)} videos.")

    # Processing settings
    frame_interval = config.get("frame_interval", 15)
    train_ratio = config.get("train_ratio", 0.8)
    text_prompt = config.get("text_prompt", "vehicle")
    box_threshold = config.get("box_threshold", 0.35)
    text_threshold = config.get("text_threshold", 0.25)
    single_class_mode = config.get("single_class_mode", True)

    total_frames = 0

    for video_path in video_files:
        print(f"Processing {video_path}...")
        cap = cv2.VideoCapture(video_path)
        video_name = Path(video_path).stem

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # Decide split
                split = "train" if random.random() < train_ratio else "val"

                # Save image first (GroundingDINO load_image expects path)
                # Format: video_name_frame_000123.jpg
                file_name = f"{video_name}_frame_{frame_idx:06d}"
                image_filename = f"{file_name}.jpg"
                label_filename = f"{file_name}.txt"

                image_path = os.path.join(output_dir, "images", split, image_filename)
                label_path = os.path.join(output_dir, "labels", split, label_filename)

                cv2.imwrite(image_path, frame)

                # Run detection
                # We interpret the saved image to ensure format consistency
                image_source, image_tensor = load_image(image_path)

                boxes, logits, phrases = predict(
                    model=model,
                    image=image_tensor,
                    caption=text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )

                # Save labels if any detections
                if len(boxes) > 0:
                    with open(label_path, "w") as f_label:
                        for i, box in enumerate(boxes):
                            cx, cy, w, h = box.tolist()

                            class_id = 0
                            if not single_class_mode:
                                pass

                            f_label.write(
                                f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                            )

                    # Preview
                    if config.get("show_preview", False):
                        h_img, w_img, _ = frame.shape
                        preview_img = frame.copy()
                        for box in boxes:
                            cx, cy, w, h = box.tolist()
                            x1 = int((cx - w / 2) * w_img)
                            y1 = int((cy - h / 2) * h_img)
                            x2 = int((cx + w / 2) * w_img)
                            y2 = int((cy + h / 2) * h_img)
                            cv2.rectangle(
                                preview_img, (x1, y1), (x2, y2), (0, 255, 0), 2
                            )

                        try:
                            cv2.imshow("Preview", preview_img)
                            cv2.waitKey(1)
                        except cv2.error:
                            print(
                                "\nWarning: cv2.imshow not supported (headless mode?). Disabling preview."
                            )
                            config["show_preview"] = False

                    total_frames += 1
                    if total_frames % 10 == 0:
                        print(f"  Saved {total_frames} annotated frames...", end="\r")
                else:
                    Path(label_path).touch()
                    if config.get("show_preview", False):
                        try:
                            cv2.imshow("Preview", frame)
                            cv2.waitKey(1)
                        except cv2.error:
                            print(
                                "\nWarning: cv2.imshow not supported (headless mode?). Disabling preview."
                            )
                            config["show_preview"] = False

            frame_idx += 1

        cap.release()
        print(f"\nFinished {video_path}")

    # Generate data.yaml
    class_names = config.get("class_names", ["vehicle"])
    data_yaml = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
    }

    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        yaml.dump(data_yaml, f)

    print("\nDataset generation complete!")
    print(f"Total labeled frames: {total_frames}")
    print(f"Dataset location: {output_dir}")
    print(f"data.yaml created at {os.path.join(output_dir, 'data.yaml')}")


if __name__ == "__main__":
    main()
