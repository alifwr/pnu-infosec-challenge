import os
import torch
import cv2
import numpy as np
import glob
import supervision as sv
from groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert


def main():
    # Configuration
    config_path = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    weights_path = "weights/groundingdino_swint_ogc.pth"
    images_dir = "images"
    cropped_dir = "images_cropped"
    text_prompt = "tank"
    box_threshold = 0.35
    text_threshold = 0.25

    # Create cropped directory
    os.makedirs(cropped_dir, exist_ok=True)

    # Check if weights exist, if not download
    if not os.path.exists(weights_path):
        print("Grounding DINO weights not found. Please download manually:")
        print(
            "wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        )
        print("And place it in weights/groundingdino_swint_ogc.pth")
        return

    # Load model
    print("Loading Grounding DINO model...")
    model = load_model(config_path, weights_path)

    # Get all images
    image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
    print(f"Found {len(image_files)} images to process")

    total_crops = 0

    for img_path in image_files:
        print(f"Processing {img_path}")

        # Load image
        image_source, image = load_image(img_path)

        # Run detection
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        # Convert boxes to xyxy format
        h, w, _ = image_source.shape
        boxes = boxes * torch.tensor([w, h, w, h])
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # Process detections
        detections = sv.Detections(
            xyxy=boxes,
            confidence=logits.numpy(),
            class_id=np.zeros(len(boxes), dtype=int),
        )

        # Filter detections
        detections = detections[detections.confidence > box_threshold]

        print(f"  Found {len(detections)} tank detections")

        # Load original image for cropping
        original_image = cv2.imread(img_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Extract class name from filename
        filename = os.path.basename(img_path)
        parts = filename.split("_")
        class_parts = []
        for i, part in enumerate(parts):
            if part in ["bing", "google"]:
                break
            class_parts.append(part)
        class_name = "_".join(class_parts)

        # Crop and save each detection
        for i, (box, confidence) in enumerate(
            zip(detections.xyxy, detections.confidence)
        ):
            x1, y1, x2, y2 = box.astype(int)

            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(original_image.shape[1], x2)
            y2 = min(original_image.shape[0], y2)

            # Crop
            cropped = original_image[y1:y2, x1:x2]

            # Skip if crop is too small
            if cropped.shape[0] < 32 or cropped.shape[1] < 32:
                continue

            # Create new filename
            base_name = os.path.splitext(filename)[0]
            new_filename = f"{base_name}_crop_{i}.jpg"
            new_path = os.path.join(cropped_dir, new_filename)

            # Save cropped image
            cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
            cv2.imwrite(new_path, cropped_bgr)

            total_crops += 1
            print(f"    Saved crop {i}: {new_filename}")

    print("\nPreprocessing complete!")
    print(f"Total cropped images: {total_crops}")
    print(f"Cropped images saved to: {cropped_dir}")


if __name__ == "__main__":
    main()
