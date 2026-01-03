from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
from collections import defaultdict
import numpy as np
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FlatStructureTankDataset(Dataset):
    """
    Dataset for flat structure where class is in filename
    Format: {CLASS_NAME}_{source}_{timestamp}_{id}.jpg

    Example: K1_Tank_bing_20251120_153946_011329.jpg -> class: K1_Tank
    """

    def __init__(
        self,
        images_dir,
        transform=None,
        mode="train",
        train_ratio=0.7,
        gallery_ratio=0.15,
        seed=42,
    ):
        # Use cropped images if available
        cropped_dir = images_dir + "_cropped"
        if os.path.exists(cropped_dir) and os.listdir(cropped_dir):
            print(f"Using cropped images from {cropped_dir}")
            images_dir = cropped_dir
        else:
            print(f"Using original images from {images_dir}")

        self.images_dir = images_dir
        self.transform = transform
        self.mode = mode

        # Parse all image files
        image_files = glob.glob(os.path.join(images_dir, "*.jpg"))

        # Group by class name (extract from filename)
        class_to_images = defaultdict(list)
        for img_path in image_files:
            filename = os.path.basename(img_path)
            # Extract class name: everything before the first timestamp pattern
            # K1_Tank_bing_20251120... -> K1_Tank
            parts = filename.split("_")

            # Find where the source (bing/google) starts
            class_parts = []
            for i, part in enumerate(parts):
                if part in ["bing", "google", "wanto"]:
                    break
                class_parts.append(part)

            class_name = "_".join(class_parts)
            class_to_images[class_name].append(img_path)

        # Create train/gallery/test splits
        np.random.seed(seed)
        self.samples = []

        for class_name, images in class_to_images.items():
            # Shuffle images
            images = sorted(images)  # Sort for reproducibility
            np.random.shuffle(images)

            n = len(images)
            n_train = int(n * train_ratio)
            n_gallery = int(n * gallery_ratio)

            if mode == "train":
                selected = images[:n_train]
            elif mode == "gallery":
                selected = images[n_train : n_train + n_gallery]
            elif mode == "test":
                selected = images[n_train + n_gallery :]
            else:
                selected = images  # Use all

            for img_path in selected:
                self.samples.append({"image_path": img_path, "label": class_name})

        # Build label mapping
        self.tank_types = sorted(list(set([s["label"] for s in self.samples])))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.tank_types)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        # Add label indices to samples
        for sample in self.samples:
            sample["label_idx"] = self.label_to_idx[sample["label"]]

        print(f"\n{'=' * 60}")
        print(f"Mode: {mode.upper()}")
        print(f"Total samples: {len(self.samples)}")
        print(f"Classes: {len(self.tank_types)}")
        print(f"{'=' * 60}")
        self._print_class_distribution()

    def _print_class_distribution(self):
        """Print samples per class"""
        class_counts = defaultdict(int)
        for sample in self.samples:
            class_counts[sample["label"]] += 1

        print("\nClass distribution:")
        for label in sorted(class_counts.keys()):
            count = class_counts[label]
            print(f"  {label:30s}: {count:3d} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")

        # Apply transforms
        if self.transform:
            image = np.array(image)
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            image = transforms.ToTensor()(image)

        label_idx = sample["label_idx"]

        return image, label_idx


def get_train_transforms(image_size=224):
    """Training augmentations - strong for few-shot"""
    transforms_list = [
        A.Resize(image_size, image_size),
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.2, rotate_limit=15, border_mode=0, p=0.7
        ),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        # Color augmentations (important for different lighting/weather)
        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0
                ),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
            ],
            p=0.6,
        ),
        # Weather/quality augmentations (simulate different conditions)
        A.OneOf(
            [
                A.GaussNoise(std_range=(0.1, 0.2), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
            ],
            p=0.4,
        ),
        # Random erasing (simulate occlusion)
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill=0,
            p=0.3,
        ),
        # Normalize
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(transforms_list)


def get_val_transforms(image_size=224):
    """Validation/Gallery transforms - minimal"""
    transforms_list = [
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(transforms_list)


# Analysis function to understand your dataset better
def analyze_dataset(images_dir):
    """Analyze the dataset structure"""
    image_files = glob.glob(os.path.join(images_dir, "*.jpg"))

    class_to_images = defaultdict(list)
    sources = defaultdict(int)

    for img_path in image_files:
        filename = os.path.basename(img_path)
        parts = filename.split("_")

        # Extract class name
        class_parts = []
        source = None
        for i, part in enumerate(parts):
            if part in ["bing", "google", "wanto"]:
                source = part
                break
            class_parts.append(part)

        class_name = "_".join(class_parts)
        class_to_images[class_name].append(img_path)

        if source:
            sources[source] += 1

    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)

    print(f"\nTotal images: {len(image_files)}")
    print(f"Total classes: {len(class_to_images)}")

    print("\nImages per class:")
    for class_name in sorted(class_to_images.keys()):
        count = len(class_to_images[class_name])
        print(f"  {class_name:30s}: {count:3d} images")

    print("\nImages by source:")
    for source, count in sorted(sources.items()):
        print(f"  {source:10s}: {count:3d} images")

    print("\nRecommended splits (70/15/15):")
    for class_name in sorted(class_to_images.keys()):
        n = len(class_to_images[class_name])
        n_train = int(n * 0.7)
        n_gallery = int(n * 0.15)
        n_test = n - n_train - n_gallery
        print(
            f"  {class_name:30s}: {n_train:2d} train, {n_gallery:2d} gallery, {n_test:2d} test"
        )

    print("=" * 60)


# Usage example
if __name__ == "__main__":
    images_dir = "data/scraped_dataset/images"

    # First, analyze the dataset
    analyze_dataset(images_dir)

    # Create datasets
    print("\n\nCreating datasets...")

    train_dataset = FlatStructureTankDataset(
        images_dir=images_dir,
        transform=get_train_transforms(224),
        mode="train",
        train_ratio=0.7,
        gallery_ratio=0.15,
    )

    gallery_dataset = FlatStructureTankDataset(
        images_dir=images_dir,
        transform=get_val_transforms(224),
        mode="gallery",
        train_ratio=0.7,
        gallery_ratio=0.15,
    )

    test_dataset = FlatStructureTankDataset(
        images_dir=images_dir,
        transform=get_val_transforms(224),
        mode="test",
        train_ratio=0.7,
        gallery_ratio=0.15,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,  # Smaller batch for few-shot
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    gallery_loader = DataLoader(
        gallery_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
    )

    print("\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Gallery batches: {len(gallery_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Test loading a batch
    print("\nTesting batch loading...")
    images, labels = next(iter(train_loader))
    print(f"  Batch shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(
        f"  Label examples: {[train_dataset.idx_to_label[lbl.item()] for lbl in labels[:3]]}"
    )
