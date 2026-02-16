# Copyright (c) 2024
# Licensed under the Apache License, Version 2.0 (the "License");
#
# MobileNetV4 backbone adapted for PaddleOCR-style usage.

from __future__ import absolute_import, division, print_function

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn import BatchNorm2D, Conv2D, ReLU, AdaptiveAvgPool2D
from paddle.nn.initializer import KaimingNormal
from paddle.regularizer import L2Decay

__all__ = ["MobileNetV4"]

block_specs_tiny = [
    # conv_bn, kernel_size, stride, out_channels
    # uib, start_dw_kernel_size, middle_dw_kernel_size, stride, out_channels, expand_ratio
    ('conv_bn', 3, 2, 32),  # 1
    ('conv_bn', 3, 2, 32),  # 2
    ('conv_bn', 1, 1, 32),
    ('conv_bn', 3, (2, 1), 96),  #
    ('conv_bn', 1, 1, 64),
    ('uib', 5, 5, (2, 1), 96, 3.0),  #
    ('uib', 0, 3, 1, 96, 2.0),
    ('uib', 0, 3, 1, 96, 2.0),
    ('uib', 0, 3, 1, 96, 2.0),
    ('uib', 0, 3, 1, 96, 2.0),
    ('uib', 0, 3, 1, 96, 2.0),
    ('uib', 3, 3, (2, 1), 128, 6.0),  #
    ('uib', 5, 5, 1, 128, 4.0),
    ('uib', 0, 5, 1, 128, 4.0),
    ('uib', 0, 5, 1, 128, 3.0),
    ('uib', 0, 3, 1, 128, 4.0),
    ('uib', 0, 3, 1, 128, 4.0),
    ('conv_bn', 1, 1, 256),
]

block_specs_small = [
    # conv_bn, kernel_size, stride, out_channels
    # uib, start_dw_kernel_size, middle_dw_kernel_size, stride, out_channels, expand_ratio
    # 112px
    ('conv_bn', 3, 2, 32),
    # 56px
    ('conv_bn', 3, 2, 32),
    ('conv_bn', 1, 1, 32),
    # 28px
    ('conv_bn', 3, 2, 96),
    ('conv_bn', 1, 1, 64),
    # 14px
    ('uib', 5, 5, 2, 96, 3.0),  # ExtraDW
    ('uib', 0, 3, 1, 96, 2.0),  # IB
    ('uib', 0, 3, 1, 96, 2.0),  # IB
    ('uib', 0, 3, 1, 96, 2.0),  # IB
    ('uib', 0, 3, 1, 96, 2.0),  # IB
    ('uib', 3, 0, 1, 96, 4.0),  # ConvNext
    # 7px
    ('uib', 3, 3, 2, 128, 6.0),  # ExtraDW
    ('uib', 5, 5, 1, 128, 4.0),  # ExtraDW
    ('uib', 0, 5, 1, 128, 4.0),  # IB
    ('uib', 0, 5, 1, 128, 3.0),  # IB
    ('uib', 0, 3, 1, 128, 4.0),  # IB
    ('uib', 0, 3, 1, 128, 4.0),  # IB
    # ('conv_bn', 1, 1, 960),  # Conv
]
block_specs_medium = [
    ('conv_bn', 3, 2, 32),
    ('conv_bn', 3, 2, 128),
    ('conv_bn', 1, 1, 48),
    # 3rd stage
    ('uib', 3, 5, 2, 80, 4.0),
    ('uib', 3, 3, 1, 80, 2.0),
    # 4th stage
    ('uib', 3, 5, 2, 160, 6.0),
    ('uib', 3, 3, 1, 160, 4.0),
    ('uib', 3, 3, 1, 160, 4.0),
    ('uib', 3, 5, 1, 160, 4.0),
    ('uib', 3, 3, 1, 160, 4.0),
    ('uib', 3, 0, 1, 160, 4.0),
    ('uib', 0, 0, 1, 160, 2.0),
    ('uib', 3, 0, 1, 160, 4.0),
    # 5th stage
    ('uib', 5, 5, 2, 256, 6.0),
    ('uib', 5, 5, 1, 256, 4.0),
    ('uib', 3, 5, 1, 256, 4.0),
    ('uib', 3, 5, 1, 256, 4.0),
    ('uib', 0, 0, 1, 256, 4.0),
    ('uib', 3, 0, 1, 256, 4.0),
    ('uib', 3, 5, 1, 256, 2.0),
    ('uib', 5, 5, 1, 256, 4.0),
    ('uib', 0, 0, 1, 256, 4.0),
    ('uib', 0, 0, 1, 256, 4.0),
    ('uib', 5, 0, 1, 256, 2.0),
    # FC layers
    # ('conv_bn', 1, 1, 960),
]
block_specs_large = [
    ("conv_bn", 3, 2, 24),
    ("conv_bn", 3, 2, 96),
    ("conv_bn", 1, 1, 48),
    ("uib", 3, 5, 2, 96, 4.0),
    ("uib", 3, 3, 1, 96, 4.0),
    ("uib", 3, 5, 2, 192, 4.0),
    ("uib", 3, 3, 1, 192, 4.0),
    ("uib", 3, 3, 1, 192, 4.0),
    ("uib", 3, 3, 1, 192, 4.0),
    ("uib", 3, 5, 1, 192, 4.0),
    ("uib", 5, 3, 1, 192, 4.0),
    ("uib", 5, 3, 1, 192, 4.0),
    ("uib", 5, 3, 1, 192, 4.0),
    ("uib", 5, 3, 1, 192, 4.0),
    ("uib", 5, 3, 1, 192, 4.0),
    ("uib", 3, 0, 1, 192, 4.0),
    ("uib", 5, 5, 2, 512, 4.0),
    ("uib", 5, 5, 1, 512, 4.0),
    ("uib", 5, 5, 1, 512, 4.0),
    ("uib", 5, 5, 1, 512, 4.0),
    ("uib", 5, 0, 1, 512, 4.0),
    ("uib", 5, 3, 1, 512, 4.0),
    ("uib", 5, 0, 1, 512, 4.0),
    ("uib", 5, 0, 1, 512, 4.0),
    ("uib", 5, 3, 1, 512, 4.0),
    ("uib", 5, 5, 1, 512, 4.0),
    ("uib", 5, 0, 1, 512, 4.0),
    ("uib", 5, 0, 1, 512, 4.0),
    ("uib", 5, 0, 1, 512, 4.0),
    # ("conv_bn", 1, 1, 960),
]


def make_divisible(value, divisor, min_value=None, round_down_protect=True):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return new_value


class ConvBNAct(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, act=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
            bias_attr=False,
        )
        self.bn = BatchNorm2D(
            out_channels,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)),
        )
        self.act = ReLU() if act else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class UniversalInvertedBottleneck(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            expand_ratio,
            start_dw_kernel_size,
            middle_dw_kernel_size,
            stride,
            middle_dw_downsample=True,
            use_layer_scale=False,
            layer_scale_init_value=1e-5,
    ):
        super().__init__()
        self.start_dw_kernel_size = start_dw_kernel_size
        self.middle_dw_kernel_size = middle_dw_kernel_size
        self.use_layer_scale = use_layer_scale

        if start_dw_kernel_size:
            start_stride = stride if not middle_dw_downsample else 1
            self.start_dw_conv = ConvBNAct(
                in_channels,
                in_channels,
                start_dw_kernel_size,
                stride=start_stride,
                groups=in_channels,
                act=False,
            )
        else:
            self.start_dw_conv = None

        expand_channels = make_divisible(in_channels * expand_ratio, 8)
        self.expand_conv = ConvBNAct(in_channels, expand_channels, 1, act=True)

        if middle_dw_kernel_size:
            middle_stride = stride if middle_dw_downsample else 1
            self.middle_dw_conv = ConvBNAct(
                expand_channels,
                expand_channels,
                middle_dw_kernel_size,
                stride=middle_stride,
                groups=expand_channels,
                act=True,
            )
        else:
            self.middle_dw_conv = None

        self.proj_conv = ConvBNAct(expand_channels, out_channels, 1, act=False)

        if use_layer_scale:
            self.gamma = self.create_parameter(
                shape=[out_channels],
                default_initializer=nn.initializer.Constant(layer_scale_init_value),
            )
        else:
            self.gamma = None

        self.identity = self._is_identity(stride, in_channels, out_channels)

    @staticmethod
    def _is_identity(stride, in_channels, out_channels):
        if isinstance(stride, (list, tuple)):
            stride_is_one = stride[0] == 1 and stride[1] == 1
        else:
            stride_is_one = stride == 1
        return stride_is_one and in_channels == out_channels

    def forward(self, x):
        shortcut = x
        if self.start_dw_conv is not None:
            x = self.start_dw_conv(x)
        x = self.expand_conv(x)
        if self.middle_dw_conv is not None:
            x = self.middle_dw_conv(x)
        x = self.proj_conv(x)
        if self.gamma is not None:
            x = x * self.gamma.reshape([1, -1, 1, 1])
        return x + shortcut if self.identity else x


class MobileNetV4(nn.Layer):
    def __init__(
            self,
            in_channels=3,
            model_name="small",
            stride_overrides=None,
            use_layer_scale=False,
            layer_scale_init_value=1e-5,
            last_pool_type="max",
            last_pool_kernel_size=[3, 2],
            **kwargs,
    ):
        super().__init__()
        if model_name == "tiny":
            self.block_specs = block_specs_tiny
        elif model_name == "small":
            self.block_specs = block_specs_small
        elif model_name == "medium":
            self.block_specs = block_specs_medium
        elif model_name == "large":
            self.block_specs = block_specs_large
        else:
            raise NotImplementedError(
                "mode[" + model_name + "_model] is not implemented!"
            )
        self.stride_overrides = stride_overrides or {}

        layers = []
        c = in_channels
        for idx, (block_type, *block_cfg) in enumerate(self.block_specs):
            stride = self.stride_overrides.get(idx)
            if block_type == "conv_bn":
                k, s, f = block_cfg
                if stride is None:
                    stride = s
                layers.append(ConvBNAct(c, f, k, stride=stride, act=True))
                c = f
            elif block_type == "uib":
                start_k, middle_k, s, f, e = block_cfg
                if stride is None:
                    stride = s
                layers.append(
                    UniversalInvertedBottleneck(
                        c,
                        f,
                        e,
                        start_k,
                        middle_k,
                        stride,
                        middle_dw_downsample=True,
                        use_layer_scale=use_layer_scale,
                        layer_scale_init_value=layer_scale_init_value,
                    )
                )
                c = f
            else:
                raise NotImplementedError("Unknown block type: {}".format(block_type))

        self.features = nn.Sequential(*layers)
        if last_pool_type == "avg":
            self.pool = nn.AvgPool2D(
                kernel_size=last_pool_kernel_size,
                stride=last_pool_kernel_size,
                padding=0,
            )
        else:
            self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.out_channels = c
        # self._initialize_weights()

    def forward(self, inputs):
        x = self.features(inputs)
        x = self.pool(x)
        # if self.training:
        #     x = F.adaptive_avg_pool2d(x, [1, 40])
        # else:
        #     x = F.avg_pool2d(x, [3, 2])
        return x

    def _initialize_weights(self):
        for m in self.sublayers():
            if isinstance(m, Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                m.weight.set_value(
                    paddle.randn(m.weight.shape) * math.sqrt(2.0 / n)
                )
            elif isinstance(m, BatchNorm2D):
                if m.weight is not None:
                    m.weight.set_value(paddle.ones_like(m.weight))
                if m.bias is not None:
                    m.bias.set_value(paddle.zeros_like(m.bias))
