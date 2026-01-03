from ultralytics import YOLO
import sys
import os


def test(model_path=None):
    # Default path to the best model from the training script
    default_model_path = "task1_image_retrieval/yolo10x_cctv_custom/weights/best.pt"

    if model_path is None:
        model_path = default_model_path  # Check relative to CWD

    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist.")
        print(f"Please provide a valid path or run training first.")
        # Fallback to verify using the initial weights if just testing the script structure
        # (Though evaluating raw weights on a dataset won't give good results)
        return

    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)

        # Validate/Test
        # runs validation on the test split
        print("Running evaluation on test split...")
        metrics = model.val(
            data="dataset/cctv_dataset/train_config.yaml",
            split="test",
            project="task1_image_retrieval",
            name="yolo10x_cctv_test",
        )

        print("\nTest Results:")
        print(f"mAP50: {metrics.box.map50}")
        print(f"mAP50-95: {metrics.box.map}")

    except Exception as e:
        print(f"Testing failed: {e}")


if __name__ == "__main__":
    # Allow passing path via CLI args
    path = sys.argv[1] if len(sys.argv) > 1 else None
    test(path)
