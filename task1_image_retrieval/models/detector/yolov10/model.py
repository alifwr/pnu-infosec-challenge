import os
import torch
import torch.nn as nn
from .modules import C2f, C2fCIB, Conv, SCDown, SPPF, PSA, v10Detect, make_divisible


class YOLOv10(nn.Module):
    """YOLOv10 model implementation."""

    def __init__(self, nc=80, scale="n", weights=None, load_components=None):
        super().__init__()
        self.nc = nc

        # Scales: depth, width, max_channels
        scales = {
            "n": [0.33, 0.25, 1024],
            "s": [0.33, 0.50, 1024],
            "m": [0.67, 0.75, 768],
            "l": [1.00, 1.00, 512],
            "x": [1.00, 1.25, 512],
        }

        depth_s, width_s, max_channels = scales.get(scale, scales["n"])

        def c(ch):
            return make_divisible(min(ch, max_channels) * width_s, 8)

        def n(num):
            return max(round(num * depth_s), 1) if num > 1 else num

        # Config per scale
        # N: b8=C2f, h19=C2f, h22=C2fCIB(lk=T)
        # S: b8=C2fCIB(lk=T), h19=C2f, h22=C2fCIB(lk=T)
        # M/L/X: b8=C2fCIB(lk=F), h19=C2fCIB(lk=F), h22=C2fCIB(lk=F)

        b8_cls, b8_lk = C2f, False
        h19_cls, h19_lk = C2f, False
        h22_lk = True

        if scale == "s":
            b8_cls, b8_lk = C2fCIB, True
            h19_cls, h19_lk = C2f, False
            h22_lk = True
        elif scale in "mlx":
            b8_cls, b8_lk = C2fCIB, False
            h19_cls, h19_lk = C2fCIB, False
            h22_lk = False

        # Backbone
        # 0: Conv(64, 3, 2)
        self.b0 = Conv(3, c(64), 3, 2)
        # 1: Conv(128, 3, 2)
        self.b1 = Conv(c(64), c(128), 3, 2)
        # 2: C2f(128, True, n=3)
        self.b2 = C2f(c(128), c(128), n(3), shortcut=True)
        # 3: Conv(256, 3, 2)
        self.b3 = Conv(c(128), c(256), 3, 2)
        # 4: C2f(256, True, n=6)
        self.b4 = C2f(c(256), c(256), n(6), shortcut=True)
        # 5: SCDown(512, 3, 2)
        self.b5 = SCDown(c(256), c(512), 3, 2)
        # 6: C2f(512, True, n=6)
        self.b6 = C2f(c(512), c(512), n(6), shortcut=True)
        # 7: SCDown(1024, 3, 2)
        self.b7 = SCDown(c(512), c(1024), 3, 2)

        # 8: C2f/C2fCIB(1024, True, n=3)
        if b8_cls == C2fCIB:
            self.b8 = b8_cls(c(1024), c(1024), n(3), shortcut=True, lk=b8_lk)
        else:
            self.b8 = b8_cls(c(1024), c(1024), n(3), shortcut=True)

        # 9: SPPF(1024, 5)
        self.b9 = SPPF(c(1024), c(1024), 5)
        # 10: PSA(1024)
        self.b10 = PSA(c(1024), c(1024))

        # Head
        # 11: Upsample
        self.h11 = nn.Upsample(scale_factor=2, mode="nearest")
        # 12: Concat(b6) (P4)
        # 13: C2f(512)
        self.h13 = C2f(c(1024) + c(512), c(512), n(3), shortcut=False)

        # 14: Upsample
        self.h14 = nn.Upsample(scale_factor=2, mode="nearest")
        # 15: Concat(b4) (P3)
        # 16: C2f(256)
        self.h16 = C2f(c(512) + c(256), c(256), n(3), shortcut=False)

        # 17: Conv(256, 3, 2)
        self.h17 = Conv(c(256), c(256), 3, 2)
        # 18: Concat(h13)
        # 19: C2f/C2fCIB(512)
        # Note: yaml h19 for N/S is C2f[512]. For M/L/X is C2fCIB[512, True] -> shortcut=True?
        # Check yaml Step 203: Line 39: C2f, [512] -> shortcut default False? C2f(512) args: [c2]. shortcut defaults False.
        # Check yaml Step 319: Line 39: C2fCIB, [512, True] -> shortcut=True.
        # So h19 shortcut also changes.

        h19_shortcut = False
        if scale in "mlx":
            h19_shortcut = True

        if h19_cls == C2fCIB:
            self.h19 = h19_cls(
                c(256) + c(512), c(512), n(3), shortcut=h19_shortcut, lk=h19_lk
            )
        else:
            self.h19 = h19_cls(c(256) + c(512), c(512), n(3), shortcut=h19_shortcut)

        # 20: SCDown(512, 3, 2)
        self.h20 = SCDown(c(512), c(512), 3, 2)
        # 21: Concat(h10)
        # 22: C2fCIB(1024, True, True/False, n=3)
        self.h22 = C2fCIB(c(512) + c(1024), c(1024), n(3), shortcut=True, lk=h22_lk)

        # 23: v10Detect
        self.detect = v10Detect(nc, ch=(c(256), c(512), c(1024)))

        if weights is not None:
            self.load_weights(weights, load_components)

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
        h21 = torch.cat([h20, x10], dim=1)  # h10 is actually x10 (output of b10)
        h22 = self.h22(h21)  # P5

        return self.detect([h16, h19, h22])

    def load_weights(self, weights_path, components=None):
        """Load weights from a .pth file (state_dict with 'model.i...' keys)."""
        if not os.path.exists(weights_path):
            print(f"Weights file not found: {weights_path}")
            return

        state_dict = torch.load(weights_path, map_location="cpu")

        # If wrapped in 'model', extract it. Use state_dict directly if keys start with model.0
        if "model" in state_dict and not list(state_dict.keys())[0].startswith(
            "model."
        ):
            state_dict = state_dict["model"]
            if hasattr(state_dict, "state_dict"):
                state_dict = state_dict.state_dict()

        new_state_dict = {}

        if components is not None:
            print(f"Loading only components: {components}")

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

                    if components is not None:
                        # simple check: if any component prefix matches new_key
                        # e.g. components=['b0', 'detect']
                        if not any(
                            new_key.startswith(c + ".") or new_key == c
                            for c in components
                        ):
                            continue

                    new_state_dict[new_key] = v
                else:
                    # Some keys might be persistent buffers or otherwise.
                    pass
            else:
                # Standard key without model.i?
                pass

        # Load
        # strict=False because we might have extra keys or missing keys (e.g. anchors)
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        print(f"Weights loaded from {weights_path}")
        if missing:
            print(f"Missing keys: {len(missing)}")
            print(f"Examples: {missing[:5]}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")
            print(f"Examples: {unexpected[:5]}")


if __name__ == "__main__":
    scale = "m"
    weights_path = "/home/alif/pnu/buffer/yolo/yolo10m_weights.pth"

    # Instantiate with auto-loading
    model = YOLOv10(scale=scale, weights=weights_path)
    print(f"YOLOv10{scale} instantiated.")

    x = torch.randn(1, 3, 640, 640)
    y = model(x)
    print(f"Output type: {type(y)}")
    if isinstance(y, dict):
        print(f"One2Many Output shapes: {[yi.shape for yi in y['one2many']]}")
    else:
        print(f"Output shapes: {[yi.shape for yi in y]}")
