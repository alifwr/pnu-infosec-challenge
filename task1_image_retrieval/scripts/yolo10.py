from ultralytics import YOLO
from modules.utils import inspect_model, load_specific_weights


# 1. Initialize custom model from YAML
print("Initializing model from config...")
model = YOLO("configs/yolo10x.yaml")

# 2. Inspect the model components
inspect_model(model)

# 3. Load backbone weights (Layers 0-10) using the new generic function
backbone_layers = list(range(23))  # 0 to 10
model = load_specific_weights(
    model, "weights/yolov10x.pt", target_layers=backbone_layers
)

# Perform object detection on an image
# Note: Results will be poor since head is random
print("Running inference...")
results = model(
    "dataset/cctv_dataset/test/images/-LIVE-CCTV-GRAMEDIA_1-2025-12-23-07_36_1766450163_converted_mp4-0004_jpg.rf.2489dc9dce332ef881ed7282dda1eb4e.jpg"
)

# Display the results
results[0].show()
