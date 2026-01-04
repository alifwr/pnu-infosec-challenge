import os
import cv2
import glob
import numpy as np


def main():
    classes = ["Pickup", "Truck", "Bus", "SUV", "MPV"]
    images_dir = "data/indonesian/images"
    cropped_dir = "data/indonesian/cropped"
    output_path = "../report/figures/dataset_samples_all_classes.jpg"

    # Setup the canvas
    rows = len(classes)
    cols = 2

    target_h = 300
    gap = 20
    text_h = 40

    # We will resize images to fixed height, keeping aspect ratio for width,
    # then pad/crop to fixed width for uniform grid?
    # Or just stack them with varying widths?
    # Let's try fixed width approx.
    target_w = 400

    for i, cls in enumerate(classes):
        # Find scraped image
        raw_pattern = os.path.join(images_dir, f"*{cls}*.jpg")
        raw_files = glob.glob(raw_pattern)

        if not raw_files:
            print(f"No raw images found for {cls}")
            continue

        # Find a raw file that has a corresponding crop
        selected_raw = None
        selected_crop = None

        for raw_file in raw_files:
            raw_basename = os.path.basename(raw_file)
            raw_name_no_ext = os.path.splitext(raw_basename)[0]

            # Look for crop
            crop_pattern = os.path.join(cropped_dir, f"{raw_name_no_ext}_crop_*.jpg")
            crop_files = glob.glob(crop_pattern)

            if crop_files:
                selected_raw = raw_file
                selected_crop = crop_files[0]  # Take first crop
                break

        if not selected_raw:
            print(f"No cropped images found for {cls}")
            continue

        print(f"Processing {cls}: {selected_raw} -> {selected_crop}")

        # Load images
        img_raw = cv2.imread(selected_raw)
        img_crop = cv2.imread(selected_crop)

        # Resize raw to target height
        h, w = img_raw.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        img_raw = cv2.resize(img_raw, (new_w, target_h))
        # Crop or pad raw to target_w
        if new_w > target_w:
            start_x = (new_w - target_w) // 2
            img_raw = img_raw[:, start_x : start_x + target_w]
        else:
            pad_lr = (target_w - new_w) // 2
            img_raw = cv2.copyMakeBorder(
                img_raw,
                0,
                0,
                pad_lr,
                target_w - new_w - pad_lr,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255],
            )

        # Resize crop to target height
        h, w = img_crop.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        img_crop = cv2.resize(img_crop, (new_w, target_h))
        # Crop or pad crop to target_w
        if new_w > target_w:
            start_x = (new_w - target_w) // 2
            img_crop = img_crop[:, start_x : start_x + target_w]
        else:
            pad_lr = (target_w - new_w) // 2
            img_crop = cv2.copyMakeBorder(
                img_crop,
                0,
                0,
                pad_lr,
                target_w - new_w - pad_lr,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255],
            )

        # Create canvas for this class
        # Width = 2 images + 3 gaps
        # Height = 1 image + 2 gaps (no text header needed if we use latex captions, but let's keep it simple or remove text?
        # User wants "each class example as a single figure", latex subfigures usually categorize them.
        # Let's clean it up: No text on image, just the images side by side.

        class_canvas_w = (target_w * 2) + (gap * 3)
        class_canvas_h = target_h + (gap * 2)

        class_canvas = (
            np.ones((class_canvas_h, class_canvas_w, 3), dtype=np.uint8) * 255
        )

        y_img = gap

        x_raw = gap
        class_canvas[y_img : y_img + target_h, x_raw : x_raw + target_w] = img_raw

        x_crop = x_raw + target_w + gap
        class_canvas[y_img : y_img + target_h, x_crop : x_crop + target_w] = img_crop

        # Save individual file
        output_filename = f"sample_{cls.lower()}_combined.jpg"
        output_full_path = os.path.join(os.path.dirname(output_path), output_filename)

        os.makedirs(os.path.dirname(output_full_path), exist_ok=True)
        cv2.imwrite(output_full_path, class_canvas)
        print(f"Saved {output_full_path}")


if __name__ == "__main__":
    main()
