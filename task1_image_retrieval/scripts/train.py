from ultralytics import YOLO
from modules.utils import load_specific_weights, inspect_model
import os
import torch


def train():
    # Define paths
    config_path = "configs/yolo10x.yaml"
    weights_path = "weights/yolov10x.pt"
    data_path = "dataset/cctv_dataset/train_config.yaml"
    project_name = "task1_image_retrieval"
    run_name = "yolo10x_cctv_custom"

    # 1. Initialize model from config
    print(f"Initializing model from {config_path}...")
    model = YOLO(config_path)

    # 2. Inspect structure
    inspect_model(model)

    # 3. Load Pretrained Weights (Backbone + Neck)
    # Layers 0-22 cover backbone + neck. Layer 23 is the head.
    target_layers = list(range(23))

    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}...")
        model = load_specific_weights(model, weights_path, target_layers=target_layers)
    else:
        print(f"Warning: Weights file {weights_path} not found. Training from scratch.")

    # 4. Train
    print("Starting training...")
    try:
        results = model.train(
            data=data_path,
            epochs=50,  # Adjust epochs as needed
            imgsz=640,
            batch=8,  # Adjust batch size based on VRAM (yolov10x is large)
            device="0" if torch.cuda.is_available() else "cpu",
            project=project_name,
            name=run_name,
            exist_ok=True,  # Overwrite existing run if same name
            plots=True,
        )
        print("Training complete.")
    except Exception as e:
        print(f"Training failed: {e}")


if __name__ == "__main__":
    train()
