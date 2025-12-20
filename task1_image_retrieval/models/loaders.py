import os

from .classifier.efficientnet_v2 import EfficientNetV2
from .classifier.swin_transformer_v2 import SwinTransformerV2
from .detector.rtdetr_v2 import RTDETRV2

AVAILABLE_MODELS = ["efficientnet_v2", "swin_transformer_v2", "rtdetr_v2"]


def load_model(name: str, config_path: str, weight_path: str):
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found at {config_path}")

    if not os.path.exists(weight_path):
        raise ValueError(f"Weight file not found at {weight_path}")

    if name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model {name} is not available. Available models: {AVAILABLE_MODELS}"
        )

    if name == "efficientnet_v2":
        model = EfficientNetV2(config_path, weight_path)
    elif name == "swin_transformer_v2":
        model = SwinTransformerV2(config_path, weight_path)
    elif name == "rtdetr_v2":
        model = RTDETRV2(config_path, weight_path)

    return model
