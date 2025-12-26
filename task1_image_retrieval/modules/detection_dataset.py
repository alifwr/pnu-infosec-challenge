import torch
import os
from torch.utils.data import Dataset
from PIL import Image


class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
        )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        img_path = os.path.join(self.img_dir, img_file)
        image = Image.open(img_path).convert("RGB")

        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_file)

        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    values = list(map(float, line.strip().split()))
                    if len(values) == 5:
                        class_label, x, y, w, h = values
                        class_label = 0  # Only need a single class 'Car'
                        boxes.append([class_label, x, y, w, h])
                    elif len(values) > 5:
                        class_label = values[0]
                        poly_coords = values[1:]
                        xs = poly_coords[0::2]
                        ys = poly_coords[1::2]
                        min_x, max_x = min(xs), max(xs)
                        min_y, max_y = min(ys), max(ys)

                        w = max_x - min_x
                        h = max_y - min_y
                        x = min_x + w / 2
                        y = min_y + h / 2
                        boxes.append([class_label, x, y, w, h])

        boxes = torch.tensor(boxes, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, boxes


def yolo_collate_fn(batch):
    images = []
    bboxes = []

    for img, box in batch:
        images.append(img)
        bboxes.append(box)

    images = torch.stack(images, dim=0)
    return images, bboxes
