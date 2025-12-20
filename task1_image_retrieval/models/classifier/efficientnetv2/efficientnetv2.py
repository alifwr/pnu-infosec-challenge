import torch.nn as nn
import copy
import torch
import os
from functools import partial
from collections import OrderedDict


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride,
        groups,
        norm_layer,
        act,
        conv_layer=nn.Conv2d,
    ):
        super(ConvBNAct, self).__init__(
            conv_layer(
                in_channel,
                out_channel,
                kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_channel),
            act(),
        )


class SEUnit(nn.Module):
    def __init__(
        self,
        in_channel,
        reduction_ratio=4,
        act1=partial(nn.SiLU, inplace=True),
        act2=nn.Sigmoid,
    ):
        super(SEUnit, self).__init__()
        hidden_dim = in_channel // reduction_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(in_channel, hidden_dim, (1, 1), bias=True)
        self.fc2 = nn.Conv2d(hidden_dim, in_channel, (1, 1), bias=True)
        self.act1 = act1()
        self.act2 = act2()

    def forward(self, x):
        return x * self.act2(self.fc2(self.act1(self.fc1(self.avg_pool(x)))))


class StochasticDepth(nn.Module):
    def __init__(self, prob, mode):
        super(StochasticDepth, self).__init__()
        self.prob = prob
        self.survival = 1.0 - prob
        self.mode = mode

    def forward(self, x):
        if self.prob == 0.0 or not self.training:
            return x
        else:
            shape = [x.size(0)] + [1] * (x.ndim - 1) if self.mode == "row" else [1]
            return x * torch.empty(shape).bernoulli_(self.survival).div_(
                self.survival
            ).to(x.device)


class MBConv(nn.Module):
    def __init__(self, c, sd_prob=0.0, act_layer=nn.SiLU, norm_layer=nn.BatchNorm2d):
        super(MBConv, self).__init__()
        new_channel = c["in_ch"] * c["expand_ratio"]
        divisible = 8
        divisible_channel = max(
            divisible, (int(new_channel + divisible / 2)) // divisible * divisible
        )
        divisible_channel += divisible if divisible_channel < 0.9 * new_channel else 0
        inter_channel = divisible_channel

        block = []

        if c["expand_ratio"] == 1:
            block.append(
                (
                    "fused",
                    ConvBNAct(
                        c["in_ch"],
                        inter_channel,
                        c["kernel"],
                        c["stride"],
                        1,
                        norm_layer,
                        act_layer,
                    ),
                )
            )
        elif c["fused"]:
            block.append(
                (
                    "fused",
                    ConvBNAct(
                        c["in_ch"],
                        inter_channel,
                        c["kernel"],
                        c["stride"],
                        1,
                        norm_layer,
                        act_layer,
                    ),
                )
            )
            block.append(
                (
                    "fused_point_wise",
                    ConvBNAct(
                        inter_channel,
                        c["out_ch"],
                        1,
                        1,
                        1,
                        norm_layer,
                        nn.Identity,
                    ),
                )
            )
        else:
            block.append(
                (
                    "linear_bottleneck",
                    ConvBNAct(
                        c["in_ch"], inter_channel, 1, 1, 1, norm_layer, act_layer
                    ),
                )
            )
            block.append(
                (
                    "depth_wise",
                    ConvBNAct(
                        inter_channel,
                        inter_channel,
                        c["kernel"],
                        c["stride"],
                        inter_channel,
                        norm_layer,
                        act_layer,
                    ),
                )
            )
            block.append(("se", SEUnit(inter_channel, 4 * c["expand_ratio"])))
            block.append(
                (
                    "point_wise",
                    ConvBNAct(
                        inter_channel,
                        c["out_ch"],
                        1,
                        1,
                        1,
                        norm_layer,
                        nn.Identity,
                    ),
                )
            )

        self.block = nn.Sequential(OrderedDict(block))
        self.use_skip_connection = c["stride"] == 1 and c["in_ch"] == c["out_ch"]
        self.stochastic_path = StochasticDepth(sd_prob, "row")

    def forward(self, x):
        out = self.block(x)
        if self.use_skip_connection:
            out = x + self.stochastic_path(out)
        return out


class EfficientNetV2(nn.Module):
    def __init__(
        self,
        layer_infos,
        out_channels=1280,
        nclass=100,
        dropout=0.2,
        stochastic_depth=0.0,
        block=MBConv,
        act_layer=nn.SiLU,
        norm_layer=nn.BatchNorm2d,
        weights=None,
        load_components=None,
    ):
        super(EfficientNetV2, self).__init__()
        self.layer_infos = layer_infos
        self.norm_layer = norm_layer
        self.act = act_layer

        self.in_channel = layer_infos[0]["in_ch"]
        self.final_stage_channel = layer_infos[-1]["out_ch"]
        self.out_channels = out_channels

        self.cur_block = 0
        self.num_block = sum(stage["num_layers"] for stage in layer_infos)
        self.stochastic_depth = stochastic_depth

        self.stem = ConvBNAct(3, self.in_channel, 3, 2, 1, self.norm_layer, self.act)
        self.blocks = nn.Sequential(
            *[
                layer
                for layer_info in layer_infos
                for layer in self.make_layers(copy.copy(layer_info), block)
            ]
        )
        self.head = nn.Sequential(
            OrderedDict(
                [
                    (
                        "bottleneck",
                        ConvBNAct(
                            self.final_stage_channel,
                            out_channels,
                            1,
                            1,
                            1,
                            self.norm_layer,
                            self.act,
                        ),
                    ),
                    ("avgpool", nn.AdaptiveAvgPool2d((1, 1))),
                    ("flatten", nn.Flatten()),
                    ("dropout", nn.Dropout(p=dropout, inplace=True)),
                    (
                        "classifier",
                        nn.Linear(out_channels, nclass) if nclass else nn.Identity(),
                    ),
                ]
            )
        )

        if weights:
            self.load_weights(weights, load_components)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))

    def make_layers(self, layer_info, block):
        layers = []
        for i in range(layer_info["num_layers"]):
            layers.append(
                block(
                    layer_info,
                    sd_prob=self.stochastic_depth * (self.cur_block / self.num_block),
                )
            )
            self.cur_block += 1
            layer_info["in_ch"] = layer_info["out_ch"]
            layer_info["stride"] = 1
        return layers

    def load_weights(self, weights_path, components=None):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        print(f"Loading weights from {weights_path}...")
        checkpoint = torch.load(weights_path, map_location="cpu")

        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

        own_state = self.state_dict()
        mismatched_keys = []
        for name, param in list(state_dict.items()):
            if name in own_state:
                if param.shape != own_state[name].shape:
                    print(
                        f"Skipping mismatching key: {name} {param.shape} -> {own_state[name].shape}"
                    )
                    del state_dict[name]
                    mismatched_keys.append(name)

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
        print("Load status:", msg)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    structures = [
        # e k  s  in  out xN  se   fused
        (1, 3, 1, 24, 24, 3, False, True),
        (4, 3, 2, 24, 48, 5, False, True),
        (4, 3, 2, 48, 80, 5, False, True),
        (4, 3, 2, 80, 160, 7, True, False),
        (6, 3, 1, 160, 176, 14, True, False),
        (6, 3, 2, 176, 304, 18, True, False),
        (6, 3, 1, 304, 512, 5, True, False),
    ]

    model = EfficientNetV2(
        layer_infos=[
            {
                "expand_ratio": layer[0],
                "kernel": layer[1],
                "stride": layer[2],
                "in_ch": layer[3],
                "out_ch": layer[4],
                "num_layers": layer[5],
                "se": layer[6],
                "fused": layer[7],
            }
            for layer in structures
        ],
        out_channels=1280,
        nclass=100,
        dropout=0.2,
        stochastic_depth=0.0,
        block=MBConv,
        act_layer=nn.SiLU,
        norm_layer=nn.BatchNorm2d,
        weights="./weights/efficientnet_v2_m_in21k_cifar100.pth",
        load_components=None,
    ).to(device)

    def get_params(m):
        return sum(p.numel() for p in m.parameters())

    print(f"Stem parameters: {get_params(model.stem):,}")
    print(f"Backbone Blocks parameters: {get_params(model.blocks):,}")
    print(f"Head parameters: {get_params(model.head):,}")
    print(f"Total parameters: {get_params(model):,}")

    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
