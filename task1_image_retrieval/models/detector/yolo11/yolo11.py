import torch
import torch.nn as nn

from .utils import make_divisible
from .basic import Conv, SPPF
from .c3k import C3k2
from .attention import C2PSA
from .head import Detect


class YOLO11(nn.Module):
    def __init__(self, nc=80, scale="n"):
        super().__init__()
        self.nc = nc

        scales = {
            "n": [0.50, 0.25, 1024],
            "s": [0.50, 0.50, 1024],
            "m": [0.50, 1.00, 512],
            "l": [1.00, 1.00, 512],
            "x": [1.00, 1.50, 512],
        }

        depth_s, width_s, max_channels = scales.get(scale, scales["n"])

        def c(ch):
            return make_divisible(min(ch, max_channels) * width_s, 8)

        def n(num):
            return max(round(num * depth_s), 1) if num > 1 else num

        # C3k logic: True for M/L/X
        c3k = scale in "mlx"

        # Backbone
        # 0: Conv(64, 3, 2)
        self.b0 = Conv(3, c(64), 3, 2)
        # 1: Conv(128, 3, 2)
        self.b1 = Conv(c(64), c(128), 3, 2)
        # 2: C3k2(256, 2, False, 0.25)
        self.b2 = C3k2(c(128), c(256), n(2), c3k=c3k, e=0.25)
        # 3: Conv(256, 3, 2)
        self.b3 = Conv(c(256), c(256), 3, 2)
        # 4: C3k2(512, 2, False, 0.25)
        self.b4 = C3k2(c(256), c(512), n(2), c3k=c3k, e=0.25)
        # 5: Conv(512, 3, 2)
        self.b5 = Conv(c(512), c(512), 3, 2)
        # 6: C3k2(512, 2, True)
        self.b6 = C3k2(c(512), c(512), n(2), c3k=True)
        # 7: Conv(1024, 3, 2)
        self.b7 = Conv(c(512), c(1024), 3, 2)
        # 8: C3k2(1024, 2, True)
        self.b8 = C3k2(c(1024), c(1024), n(2), c3k=True)
        # 9: SPPF(1024, 5)
        self.b9 = SPPF(c(1024), c(1024), 5)
        # 10: C2PSA(1024)
        self.b10 = C2PSA(c(1024), c(1024))

        # Head
        # 11: Upsample
        self.h11 = nn.Upsample(scale_factor=2, mode="nearest")
        # 12: Concat(b6) (P4)
        # 13: C3k2(512, 2, False)
        self.h13 = C3k2(
            c(1024) + c(512), c(512), n(2), c3k=c3k
        )  # Input is b10(up) + b6

        # 14: Upsample
        self.h14 = nn.Upsample(scale_factor=2, mode="nearest")
        # 15: Concat(b4) (P3)
        # 16: C3k2(256, 2, False)
        self.h16 = C3k2(
            c(512) + c(512), c(256), n(2), c3k=c3k
        )  # Input is h13(up) + b4.

        # 17: Conv(256, 3, 2)
        self.h17 = Conv(c(256), c(256), 3, 2)
        # 18: Concat(h13)
        # 19: C3k2(512, 2, False)
        self.h19 = C3k2(c(256) + c(512), c(512), n(2), c3k=c3k)

        # 20: Conv(512, 3, 2)
        self.h20 = Conv(c(512), c(512), 3, 2)
        # 21: Concat(h10)
        # 22: C3k2(1024, 2, True)
        self.h22 = C3k2(c(512) + c(1024), c(1024), n(2), c3k=True)

        # 23: Detect
        self.detect = Detect(nc, ch=(c(256), c(512), c(1024)))

    def forward(self, x):
        # Backbone
        x0 = self.b0(x)
        x1 = self.b1(x0)
        x2 = self.b2(x1)  # P2
        x3 = self.b3(x2)
        x4 = self.b4(x3)  # P3
        x5 = self.b5(x4)
        x6 = self.b6(x5)  # P4
        x7 = self.b7(x6)
        x8 = self.b8(x7)
        x9 = self.b9(x8)
        x10 = self.b10(x9)  # P5

        # Head
        h11 = self.h11(x10)
        h12 = torch.cat([h11, x6], dim=1)
        h13 = self.h13(h12)  # P4

        h14 = self.h14(h13)
        h15 = torch.cat([h14, x4], dim=1)
        h16 = self.h16(h15)  # P3

        h17 = self.h17(h16)
        h18 = torch.cat([h17, h13], dim=1)
        h19 = self.h19(h18)  # P4

        h20 = self.h20(h19)
        h21 = torch.cat([h20, x10], dim=1)
        h22 = self.h22(h21)  # P5

        # Detect
        return self.detect([h16, h19, h22])

    def load_weights(self, weights_path):
        """Load weights from a .pth file (state_dict with 'model.i...' keys)."""
        state_dict = torch.load(weights_path, map_location="cpu")

        # If wrapped in 'model', extract it. Use state_dict directly if keys start with model.0
        if "model" in state_dict and not list(state_dict.keys())[0].startswith(
            "model."
        ):
            state_dict = state_dict["model"]
            if hasattr(state_dict, "state_dict"):
                state_dict = state_dict.state_dict()

        new_state_dict = {}
        for k, v in state_dict.items():
            # key example: model.0.conv.weight
            parts = k.split(".")
            if parts[0] == "model":
                idx = int(parts[1])
                rest = ".".join(parts[2:])

                if 0 <= idx <= 10:
                    target_attr = f"b{idx}"
                elif 11 <= idx <= 22:
                    target_attr = f"h{idx}"
                elif idx == 23:
                    target_attr = "detect"
                else:
                    print(f"Warning: Unexpected layer index {idx} in {k}")
                    continue

                # Check if this attribute exists in our model
                if hasattr(self, target_attr):
                    new_key = f"{target_attr}.{rest}"
                    new_state_dict[new_key] = v
                else:
                    pass
            else:
                pass

        # Load
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        print(f"Weights loaded from {weights_path}")
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")


if __name__ == "__main__":
    # Test loading 'm' model
    model = YOLO11(scale="m")
    print("YOLO11m instantiated.")
    print(model)

    weights_path = "yolo/yolo11m_weights.pth"
    import os

    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    else:
        print(f"Weights file not found at {weights_path}, skipping load.")

    x = torch.randn(1, 3, 640, 640)
    y = model(x)
    print(f"Output shapes: {[yi.shape for yi in y]}")
