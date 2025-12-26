import os
import json

from .classifier.efficientnet_v2 import EfficientNetV2
from .classifier.swin_transformer_v2 import SwinTransformerV2
from .detector.rtdetr_v2 import RTDETRV2
from .detector.yolov10.model import YOLOv10

AVAILABLE_MODELS = [
    "efficientnet_v2",
    "swin_transformer_v2",
    "rtdetr_v2",
    "yolov10",
]


def load_model(name: str, config_path: str, weight_path: str = None):
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found at {config_path}")

    if name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model {name} is not available. Available models: {AVAILABLE_MODELS}"
        )

    if name == "efficientnet_v2":
        model = EfficientNetV2(config_path, weight_path)
    elif name == "swin_transformer_v2":
        model = SwinTransformerV2(config_path, weight_path)
    elif name == "rtdetr_v2":
        with open(config_path, "r") as f:
            config = json.load(f)

        # Override weights if provided
        weights = weight_path if weight_path else config.get("weights_path")

        model = RTDETRV2(
            num_classes=1,  # Hardcoded for this task as requested/implied by usage
            backbone_conf=config.get("backbone_conf"),
            encoder_conf=config.get("encoder_conf"),
            decoder_conf=config.get("decoder_conf"),
            weights=weights,
            load_components=["backbone", "encoder"] if weights else None,
        )

    elif name == "yolov10":
        if weight_path is None and config_path:
            with open(config_path, "r") as f:
                config = json.load(f)
            weight_path = config.get("weights_path")

        scale = "n"
        if weight_path is not None:
            fname = os.path.basename(weight_path)
            if "yolo10s" in fname:
                scale = "s"
            elif "yolo10m" in fname:
                scale = "m"
            elif "yolo10l" in fname:
                scale = "l"
            elif "yolo10x" in fname:
                scale = "x"

        # load_components: backbone only (b0-b10) as requested ("no head")
        load_components = [f"b{i}" for i in range(11)]

        model = YOLOv10(
            nc=1, scale=scale, weights=weight_path, load_components=load_components
        )

    return model
