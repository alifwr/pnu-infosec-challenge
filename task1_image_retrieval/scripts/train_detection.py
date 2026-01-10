from ultralytics import YOLO, RTDETR
from modules.utils import load_specific_weights
import os
import torch

import wandb


def train():
    # Define paths
    config_path = "configs/"
    config_files = ["rtdetr-x.yaml"]
    weights_path = "weights/"
    weights_files = ["rtdetr-x.pt"]
    data_path = "dataset/yolo_vehicles/data.yaml"
    project_name = "vehicle_detection"
    run_name = "vehicles_yolo10x"
0
    # Initialize wandb
    try:
        wandb.login()
        print("Logged into WandB successfully.")
    except Exception as e:
        print(f"WandB login skipped/failed: {e}")
        print("Set WANDB_API_KEY environment variable for automatic login.")

    # Iterate over models
    benchmark_results = []

    for config_file, weights_file in zip(config_files, weights_files):
        # Construct full paths
        full_config = os.path.join(config_path, config_file)
        full_weights = os.path.join(weights_path, weights_file)

        # Derive run name from config (e.g., "yolo10l.yaml" -> "vehicles_yolo10l")
        model_name = os.path.splitext(config_file)[0]
        run_name = f"vehicles_{model_name}"

        print("\n" + "=" * 50)
        print(f"Training Model: {model_name}")
        print(f"Config: {full_config}")
        print(f"Weights: {full_weights}")
        print("=" * 50)

        # 1. Initialize model
        try:
            if "rtdetr" in model_name.lower():
                model = RTDETR(full_config)
            else:
                model = YOLO(full_config)
        except Exception as e:
            print(f"Failed to initialize model {model_name}: {e}")
            continue

        # 2. Load Pretrained Weights
        if os.path.exists(full_weights):
            print(f"Loading weights from {full_weights}...")

            # Use custom loader for both YOLO and RTDETR, but with different layer depths
            target_layers = []
            if "rtdetr" in model_name.lower():
                # RT-DETR structure is different, usually first 26 modules cover backbone + hybrid encoder
                target_layers = list(range(32))
            elif "yolo" in model_name.lower():
                # Layers 0-22 cover backbone + neck for YOLOv10. Layer 23 is the head.
                target_layers = list(range(23))

            if target_layers:
                try:
                    model = load_specific_weights(
                        model, full_weights, target_layers=target_layers
                    )
                except Exception as e:
                    print(
                        f"Custom weight loading failed for {model_name}: {e}. Trying standard load."
                    )
                    try:
                        # Fallback to standard load
                        if "rtdetr" in model_name.lower():
                            weights_model = RTDETR(full_weights)
                            model.model.load_state_dict(
                                weights_model.model.state_dict(), strict=False
                            )
                        else:
                            model = YOLO(full_weights)
                    except Exception as e2:
                        print(f"Fallback loading failed: {e2}")

        else:
            print(
                f"Warning: Weights file {full_weights} not found. Training from scratch."
            )

        # 3. Train
        print(f"Starting training for {model_name}...")

        # Initialize WandB run manually to control config tracking
        # Ultralytics normally handles this automatically if WANDB_DISABLED is false,
        # but manual init allows cleaner separation in loops.
        try:
            wandb.init(
                project=project_name,
                name=run_name,
                config={
                    "model": model_name,
                    "epochs": 100,
                    "batch_size": 8 if "x" in model_name else 16,
                    "config_file": full_config,
                },
                reinit=True,
            )
        except Exception as e:
            print(f"Could not init wandb run: {e}")

        # Add WandB logging callback
        def on_train_epoch_end(trainer):
            if wandb.run:
                wandb.log(trainer.metrics)

        model.add_callback("on_train_epoch_end", on_train_epoch_end)

        try:
            results = model.train(
                data=data_path,
                epochs=100,
                imgsz=640,
                batch=8 if "x" in model_name else 16,  # Reduce batch for 'x' models
                device="0" if torch.cuda.is_available() else "cpu",
                project=project_name,
                name=run_name,
                exist_ok=True,
                plots=True,
            )
            print(f"Training complete for {model_name}.")

            # 4. Collect Metrics
            # Ultralytics results object has .box.map50, .box.map, and .speed
            map50 = results.box.map50
            map50_95 = results.box.map

            # Speed is usually a dict: {'preprocess': X, 'inference': Y, 'postprocess': Z} per image in ms
            inference_time = results.speed.get("inference", 0.0)

            benchmark_results.append(
                {
                    "Model": model_name,
                    "mAP@50": map50,
                    "mAP@50-95": map50_95,
                    "Inference Time (ms)": inference_time,
                    "Parameters": sum(p.numel() for p in model.parameters())
                    / 1e6,  # Million params
                }
            )

            wandb.finish()

        except Exception as e:
            print(f"Training/Benchmark failed for {model_name}: {e}")
            wandb.finish()

    # 5. Save Report
    if benchmark_results:
        import pandas as pd

        df = pd.DataFrame(benchmark_results)
        df.to_csv("benchmark_report.csv", index=False)
        print("\n" + "=" * 50)
        print("BENCHMARK REPORT")
        print("=" * 50)
        print(df.to_string())
        print(f"\nReport saved to {os.path.abspath('benchmark_report.csv')}")
    else:
        print("\nNo benchmark results collected.")


if __name__ == "__main__":
    train()
