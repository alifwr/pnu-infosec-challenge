import torch.nn as nn
import torch
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


class MBConv(nn.Module):
    def __init__(self, c, sd_prob=0.0):
        super(MBConv, self).__init__()
        inter_channel = c.adjust_channels(c.in_ch, c.expand_ratio)
        block = []

        if c.expand_ratio == 1:
            block.append(
                (
                    "fused",
                    ConvBNAct(
                        c.in_ch,
                        inter_channel,
                        c.kernel,
                        c.stride,
                        1,
                        c.norm_layer,
                        c.act,
                    ),
                )
            )
        elif c.fused:
            block.append(
                (
                    "fused",
                    ConvBNAct(
                        c.in_ch,
                        inter_channel,
                        c.kernel,
                        c.stride,
                        1,
                        c.norm_layer,
                        c.act,
                    ),
                )
            )
            block.append(
                (
                    "fused_point_wise",
                    ConvBNAct(
                        inter_channel, c.out_ch, 1, 1, 1, c.norm_layer, nn.Identity
                    ),
                )
            )
        else:
            block.append(
                (
                    "linear_bottleneck",
                    ConvBNAct(c.in_ch, inter_channel, 1, 1, 1, c.norm_layer, c.act),
                )
            )
            block.append(
                (
                    "depth_wise",
                    ConvBNAct(
                        inter_channel,
                        inter_channel,
                        c.kernel,
                        c.stride,
                        inter_channel,
                        c.norm_layer,
                        c.act,
                    ),
                )
            )
            block.append(("se", SEUnit(inter_channel, 4 * c.expand_ratio)))
            block.append(
                (
                    "point_wise",
                    ConvBNAct(
                        inter_channel, c.out_ch, 1, 1, 1, c.norm_layer, nn.Identity
                    ),
                )
            )

        self.block = nn.Sequential(OrderedDict(block))
        self.use_skip_connection = c.stride == 1 and c.in_ch == c.out_ch
        self.stochastic_path = StochasticDepth(sd_prob, "row")

    def forward(self, x):
        out = self.block(x)
        if self.use_skip_connection:
            out = x + self.stochastic_path(out)
        return out


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


class EfficientNetV2(nn.Module):
    def __init__(
        self,
        layer_infos,
        out_channels=1280,
        nclass=0,
        dropout=0.2,
        stochastic_depth=0.0,
        block=MBConv,
        act_layer=nn.SiLU,
        norm_layer=nn.BatchNorm2d,
    ):
        super(EfficientNetV2, self).__init__()
        self.layer_infos = layer_infos
        self.norm_layer = norm_layer
        self.act = act_layer

        self.in_channel = layer_infos[0].in_ch
        self.final_stage_channel = layer_infos[-1].out_ch
        self.out_channels = out_channels

        self.cur_block = 0
        self.num_block = sum(stage.num_layers for stage in layer_infos)
        self.stochastic_depth = stochastic_depth

        self.stem = ConvBNAct(3, self.in_channel, 3, 2, 1, self.norm_layer, self.act)
        self.blocks = nn.Sequential(*self.make_stages(layer_infos, block))
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
