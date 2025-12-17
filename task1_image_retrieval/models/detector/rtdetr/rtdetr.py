import torch.nn as nn
import torch
import os

from .presnet import PResNet
from .hybrid_encoder import HybridEncoder
from .rtdetr_decoder import RTDETRTransformer


class RTDETR(nn.Module):
    def __init__(
        self,
        num_classes=80,
        backbone_conf=None,
        encoder_conf=None,
        decoder_conf=None,
        weights=None,
        load_components=None,
    ):
        super().__init__()

        self.backbone_conf = (
            backbone_conf
            if backbone_conf is not None
            else {
                "depth": 50,
                "variant": "d",
                "num_stages": 4,
                "return_idx": [1, 2, 3],
                "act": "relu",
                "freeze_norm": False,
                "pretrained": False,
            }
        )

        self.encoder_conf = (
            encoder_conf
            if encoder_conf is not None
            else {
                "in_channels": [512, 1024, 2048],
                "feat_strides": [8, 16, 32],
                "hidden_dim": 256,
                "use_encoder_idx": [2],
                "num_encoder_layers": 1,
                "expansion": 1.0,
                "depth_mult": 1.0,
            }
        )

        self.decoder_conf = (
            decoder_conf
            if decoder_conf is not None
            else {
                "hidden_dim": 256,
                "feat_channels": [256, 256, 256],
                "feat_strides": [8, 16, 32],
                "num_queries": 300,
                "num_denoising": 100,
            }
        )

        self.num_classes = num_classes
        self.config = {
            "num_classes": num_classes,
            "backbone": self.backbone_conf,
            "encoder": self.encoder_conf,
            "decoder": self.decoder_conf,
        }

        self.backbone = PResNet(**self.backbone_conf)
        self.encoder = HybridEncoder(**self.encoder_conf)
        self.decoder = RTDETRTransformer(num_classes=num_classes, **self.decoder_conf)

        if weights is not None:
            self.load_weights(weights, load_components)

    def load_weights(self, weights_path, components=None):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        print(f"Loading weights from {weights_path}...")
        checkpoint = torch.load(weights_path, map_location="cpu")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "ema" in checkpoint and "module" in checkpoint["ema"]:
            state_dict = checkpoint["ema"]["module"]
        else:
            state_dict = checkpoint

        if components is not None:
            print(f"Loading only components: {components}")
            new_state_dict = {}
            for key, value in state_dict.items():
                for component in components:
                    if key.startswith(component + "."):
                        new_state_dict[key] = value
                        break
            state_dict = new_state_dict

        msg = self.load_state_dict(state_dict, strict=False)

        if components is not None:
            real_missing = [
                k for k in msg.missing_keys if any(k.startswith(c) for c in components)
            ]
            if real_missing:
                print(f"Warning: Missing keys in requested components: {real_missing}")
            else:
                print(f"Successfully loaded components: {components}")

            if msg.unexpected_keys:
                print(f"Unexpected keys: {msg.unexpected_keys}")
        else:
            print("Load status:", msg)

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self


if __name__ == "__main__":
    import torch
    import os

    custom_backbone_conf = {
        "depth": 34,
        "freeze_at": -1,
        "freeze_norm": False,
        "pretrained": True,
        "variant": "d",
        "return_idx": [1, 2, 3],
        "num_stages": 4,
    }

    custom_encoder_conf = {
        "in_channels": [128, 256, 512],
        "hidden_dim": 256,
        "expansion": 0.5,
        "use_encoder_idx": [2],
        "num_encoder_layers": 1,
        "nhead": 8,
        "dim_feedforward": 1024,
        "dropout": 0.1,
        "enc_act": "gelu",
        "depth_mult": 1.0,
        "act": "silu",
    }

    custom_decoder_conf = {
        "num_layers": 4,
        "feat_channels": [256, 256, 256],
        "feat_strides": [8, 16, 32],
        "hidden_dim": 256,
        "num_levels": 3,
        "num_queries": 300,
        "num_denoising": 100,
        "label_noise_ratio": 0.5,
        "box_noise_scale": 1.0,
        "eval_idx": -1,
        "num_points": [4, 4, 4],
        "cross_attn_method": "default",
        "query_select_method": "default",
        "eval_spatial_size": [640, 640],
    }

    print(
        "Instantiating model with custom backbone, encoder, decoder configs and loading weights..."
    )
    model = RTDETR(
        num_classes=1,
        backbone_conf=custom_backbone_conf,
        encoder_conf=custom_encoder_conf,
        decoder_conf=custom_decoder_conf,
        weights="weights/rtdetrv2_r34vd_120e_coco_ema.pth",
        load_components=["backbone", "encoder"],
    )
    model.train()

    print("\nModel Parameters:")

    def get_params(m):
        return sum(p.numel() for p in m.parameters())

    print(f"Backbone parameters: {get_params(model.backbone):,}")
    print(f"Encoder parameters: {get_params(model.encoder):,}")
    print(f"Decoder parameters: {get_params(model.decoder):,}")
    print(f"Total parameters: {get_params(model):,}")

    x = torch.randn(2, 3, 640, 640)

    targets = [
        {"labels": torch.tensor([0, 0], dtype=torch.long), "boxes": torch.rand(2, 4)}
        for _ in range(2)
    ]

    print("\nRunning forward pass...")
    # model.eval()
    output = model(x, targets)

    print("\nOutput keys:", output.keys())
    if "pred_logits" in output:
        print("pred_logits:", output["pred_logits"].shape)
    if "pred_boxes" in output:
        print("pred_boxes:", output["pred_boxes"].shape)
