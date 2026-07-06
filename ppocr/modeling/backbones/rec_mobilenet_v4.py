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
from paddle.nn import BatchNorm2D, Conv2D, ReLU, AdaptiveAvgPool2D, LayerNorm
from paddle.nn.initializer import KaimingNormal
from paddle.regularizer import L2Decay

__all__ = ["MobileNetV4"]

# ============================================================================
# Block Specifications for Different Model Variants
# ============================================================================

block_specs_tiny = [
    # conv_bn, kernel_size, stride, out_channels
    # uib, start_dw_kernel_size, middle_dw_kernel_size, middle_dw_downsample, stride, out_channels, expand_ratio, use_layer_scale
    # input: (32, 128)
    # P1/2 16, 64
    ('conv_bn', 3, 2, 16),
    # P2/4 16, 64
    ('conv_bn', 3, 1, 48),
    ('conv_bn', 1, 1, 32),
    # P3/8 8, 64
    ('uib', 3, 3, True, (2, 1), 48, 3.0, False),  # 3. ExtraDW
    ('uib', 0, 3, True, 1, 48, 2.0, False),  # IB
    ('uib', 0, 3, True, 1, 48, 2.0, False),  # IB
    ('uib', 3, 0, True, 1, 48, 4.0, False),  # ConvNext
    # P4/16 4, 64
    ('uib', 3, 3, True, (2, 1), 96, 6.0, False),  # 7. ExtraDW
    ('uib', 0, 3, True, 1, 96, 4.0, False),  # IB
    ('uib', 0, 3, True, 1, 96, 4.0, False),  # IB
    # # P5/32 2, 32
    ('conv_bn', 3, (2, 1), 960),
    ('conv_bn', 1, 1, 512),  # Conv
]

block_specs_small = [
    # conv_bn, kernel_size, stride, out_channels
    # uib, start_dw_kernel_size, middle_dw_kernel_size, middle_dw_downsample, stride, out_channels, expand_ratio, use_layer_scale
    # P1/2
    ('conv_bn', 3, 2, 32),
    # P2/4
    ('conv_bn', 3, (2, 1), 32),
    ('conv_bn', 1, 1, 32),
    # P3/8
    ('conv_bn', 3, (2, 1), 96),
    ('conv_bn', 1, 1, 64),
    # P4/16
    ('uib', 5, 5, True, (2, 1), 96, 3.0, False),  # ExtraDW
    ('uib', 0, 3, True, 1, 96, 2.0, False),  # IB
    ('uib', 0, 3, True, 1, 96, 2.0, False),  # IB
    ('uib', 0, 3, True, 1, 96, 2.0, False),  # IB
    ('uib', 0, 3, True, 1, 96, 2.0, False),  # IB
    ('uib', 3, 0, True, 1, 96, 4.0, False),  # ConvNext
    # P5/32
    ('uib', 3, 3, True, (2, 1), 128, 6.0, False),  # ExtraDW
    ('uib', 5, 5, True, 1, 128, 4.0, False),  # ExtraDW
    ('uib', 0, 5, True, 1, 128, 4.0, False),  # IB
    ('uib', 0, 5, True, 1, 128, 3.0, False),  # IB
    ('uib', 0, 3, True, 1, 128, 4.0, False),  # IB
    ('uib', 0, 3, True, 1, 128, 4.0, False),  # IB
    ('conv_bn', 1, 1, 960),  # Conv
]

block_specs_medium = [
    ('conv_bn', 3, 2, 32),
    ('conv_bn', 3, (2, 1), 128),
    ('conv_bn', 1, 1, 48),
    # 3rd stage
    ('uib', 3, 5, True, (2, 1), 80, 4.0, False),
    ('uib', 3, 3, True, 1, 80, 2.0, False),
    # 4th stage
    ('uib', 3, 5, True, (2, 1), 160, 6.0, False),
    ('uib', 3, 3, True, 1, 160, 4.0, False),
    ('uib', 3, 3, True, 1, 160, 4.0, False),
    ('uib', 3, 5, True, 1, 160, 4.0, False),
    ('uib', 3, 3, True, 1, 160, 4.0, False),
    ('uib', 3, 0, True, 1, 160, 4.0, False),
    ('uib', 0, 0, True, 1, 160, 2.0, False),
    ('uib', 3, 0, True, 1, 160, 4.0, False),
    # 5th stage
    ('uib', 5, 5, True, (2, 1), 256, 6.0, False),
    ('uib', 5, 5, True, 1, 256, 4.0, False),
    ('uib', 3, 5, True, 1, 256, 4.0, False),
    ('uib', 3, 5, True, 1, 256, 4.0, False),
    ('uib', 0, 0, True, 1, 256, 4.0, False),
    ('uib', 3, 0, True, 1, 256, 4.0, False),
    ('uib', 3, 5, True, 1, 256, 2.0, False),
    ('uib', 5, 5, True, 1, 256, 4.0, False),
    ('uib', 0, 0, True, 1, 256, 4.0, False),
    ('uib', 0, 0, True, 1, 256, 4.0, False),
    ('uib', 5, 0, True, 1, 256, 2.0, False),
    # FC layers
    ('conv_bn', 1, 1, 960),
]

block_specs_large = [
    ("conv_bn", 3, 2, 24),
    ("conv_bn", 3, (2, 1), 96),
    ("conv_bn", 1, 1, 48),
    # 3rd stage
    ("uib", 3, 5, True, (2, 1), 96, 4.0, False),
    ("uib", 3, 3, True, 1, 96, 4.0, False),
    # 4th stage
    ("uib", 3, 5, True, (2, 1), 192, 4.0, False),
    ("uib", 3, 3, True, 1, 192, 4.0, False),
    ("uib", 3, 3, True, 1, 192, 4.0, False),
    ("uib", 3, 3, True, 1, 192, 4.0, False),
    ("uib", 3, 5, True, 1, 192, 4.0, False),
    ("uib", 5, 3, True, 1, 192, 4.0, False),
    ("uib", 5, 3, True, 1, 192, 4.0, False),
    ("uib", 5, 3, True, 1, 192, 4.0, False),
    ("uib", 5, 3, True, 1, 192, 4.0, False),
    ("uib", 5, 3, True, 1, 192, 4.0, False),
    ("uib", 3, 0, True, 1, 192, 4.0, False),
    # 5th stage
    ("uib", 5, 5, True, (2, 1), 512, 4.0, False),
    ("uib", 5, 5, True, 1, 512, 4.0, False),
    ("uib", 5, 5, True, 1, 512, 4.0, False),
    ("uib", 5, 5, True, 1, 512, 4.0, False),
    ("uib", 5, 0, True, 1, 512, 4.0, False),
    ("uib", 5, 3, True, 1, 512, 4.0, False),
    ("uib", 5, 0, True, 1, 512, 4.0, False),
    ("uib", 5, 0, True, 1, 512, 4.0, False),
    ("uib", 5, 3, True, 1, 512, 4.0, False),
    ("uib", 5, 5, True, 1, 512, 4.0, False),
    ("uib", 5, 0, True, 1, 512, 4.0, False),
    ("uib", 5, 0, True, 1, 512, 4.0, False),
    ("uib", 5, 0, True, 1, 512, 4.0, False),
    # FC layers
    ("conv_bn", 1, 1, 960),
]


# ============================================================================
# Configuration Helpers
# ============================================================================

def create_mhsa_config(num_heads, key_dim, value_dim, feature_height):
    """Create Multi-Head Self-Attention configuration.

    Args:
        num_heads: Number of attention heads
        key_dim: Dimension per head for keys/queries
        value_dim: Dimension per head for values
        feature_height: Spatial height of feature map (e.g., 24, 12, 6)

    Returns:
        List of MHSA parameters: [num_heads, key_dim, value_dim, query_h_strides,
                                   query_w_strides, kv_strides, use_layer_scale,
                                   use_multi_query, use_residual]
    """
    # Determine key-value stride based on feature map size for efficient downsampling
    if feature_height >= 24:
        kv_strides = 2
    elif feature_height >= 12:
        kv_strides = 1
    else:
        kv_strides = 1

    query_h_strides = 1
    query_w_strides = 1
    use_layer_scale = True
    use_multi_query = True
    use_residual = True

    return [
        num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides,
        use_layer_scale, use_multi_query, use_residual
    ]


# ============================================================================
# Utility Functions
# ============================================================================

def create_hybrid_block_spec(base_spec, mhsa_block_indices):
    """Create a hybrid variant by adding MHSA blocks and enabling layer scaling.
    
    Args:
        base_spec: Base block specification list
        mhsa_block_indices: Dict mapping block indices to MHSA config tuples
                           e.g., {10: create_mhsa_config(4, 64, 64, 24)}
    
    Returns:
        Modified block specification with MHSA blocks and layer scaling enabled
    """
    hybrid_spec = []
    for block_idx, block in enumerate(base_spec):
        if block[0] == 'uib':
            # Enable layer_scale for all UIB blocks in hybrid variants
            block_list = list(block)
            if len(block_list) >= 8:
                block_list[7] = True  # use_layer_scale = True
            else:
                block_list.append(True)  # Add use_layer_scale = True

            # Add MHSA config if specified for this block index
            if block_idx in mhsa_block_indices:
                block_list.append(mhsa_block_indices[block_idx])

            hybrid_spec.append(tuple(block_list))
        else:
            hybrid_spec.append(block)

    return hybrid_spec


def make_divisible(value, divisor, min_value=None, round_down_protect=True):
    """Make a value divisible by a given divisor.
    
    Rounds the value to the nearest multiple of divisor, useful for ensuring
    channel counts are divisible by group sizes or optimization requirements.
    
    Args:
        value: The value to round
        divisor: The divisor to make value divisible by
        min_value: Minimum value threshold (defaults to divisor)
        round_down_protect: Prevent rounding down by more than 10%
    
    Returns:
        Value rounded to nearest multiple of divisor
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return new_value


# ============================================================================
# Hybrid Block Specifications
# ============================================================================

# Hybrid variants: built from base specs with MHSA attention blocks
block_specs_hybrid_tiny = create_hybrid_block_spec(
    block_specs_tiny,
    mhsa_block_indices={
        # Stage 3 (48 channels, height ~8)
        4: create_mhsa_config(num_heads=4, key_dim=24, value_dim=24, feature_height=8),
        5: create_mhsa_config(num_heads=4, key_dim=24, value_dim=24, feature_height=8),
        6: create_mhsa_config(num_heads=4, key_dim=24, value_dim=24, feature_height=8),
        # Stage 4 (96 channels, height ~4)
        8: create_mhsa_config(num_heads=4, key_dim=24, value_dim=24, feature_height=4),
        9: create_mhsa_config(num_heads=4, key_dim=24, value_dim=24, feature_height=4),
    }
)

block_specs_hybrid_small = create_hybrid_block_spec(
    block_specs_small,
    mhsa_block_indices={
        # Stage 3 (96 channels, height ~14): Last 3 UIB blocks with MHSA
        8: create_mhsa_config(num_heads=4, key_dim=48, value_dim=48, feature_height=14),
        9: create_mhsa_config(num_heads=4, key_dim=48, value_dim=48, feature_height=14),
        10: create_mhsa_config(num_heads=4, key_dim=48, value_dim=48, feature_height=14),
        # Stage 4 (128 channels, height ~7): Last 3 UIB blocks with MHSA
        14: create_mhsa_config(num_heads=4, key_dim=48, value_dim=48, feature_height=7),
        15: create_mhsa_config(num_heads=4, key_dim=48, value_dim=48, feature_height=7),
        16: create_mhsa_config(num_heads=4, key_dim=48, value_dim=48, feature_height=7),
    }
)

block_specs_hybrid_medium = create_hybrid_block_spec(
    block_specs_medium,
    mhsa_block_indices={
        # Stage 3 (160 channels, height ~24): Last 4 UIB blocks with MHSA
        10: create_mhsa_config(num_heads=4, key_dim=64, value_dim=64, feature_height=24),
        11: create_mhsa_config(num_heads=4, key_dim=64, value_dim=64, feature_height=24),
        12: create_mhsa_config(num_heads=4, key_dim=64, value_dim=64, feature_height=24),
        13: create_mhsa_config(num_heads=4, key_dim=64, value_dim=64, feature_height=24),
        # Stage 4 (256 channels, height ~12): Last 3 UIB blocks with MHSA
        21: create_mhsa_config(num_heads=4, key_dim=64, value_dim=64, feature_height=12),
        22: create_mhsa_config(num_heads=4, key_dim=64, value_dim=64, feature_height=12),
        23: create_mhsa_config(num_heads=4, key_dim=64, value_dim=64, feature_height=12),
    }
)

block_specs_hybrid_large = create_hybrid_block_spec(
    block_specs_large,
    mhsa_block_indices={
        # Stage 3 (192 channels, height ~24): Last 4 UIB blocks with MHSA
        11: create_mhsa_config(num_heads=8, key_dim=48, value_dim=48, feature_height=24),
        12: create_mhsa_config(num_heads=8, key_dim=48, value_dim=48, feature_height=24),
        13: create_mhsa_config(num_heads=8, key_dim=48, value_dim=48, feature_height=24),
        14: create_mhsa_config(num_heads=8, key_dim=48, value_dim=48, feature_height=24),
        # Stage 4 (512 channels, height ~12): Last 4 UIB blocks with MHSA
        24: create_mhsa_config(num_heads=8, key_dim=64, value_dim=64, feature_height=12),
        25: create_mhsa_config(num_heads=8, key_dim=64, value_dim=64, feature_height=12),
        26: create_mhsa_config(num_heads=8, key_dim=64, value_dim=64, feature_height=12),
        27: create_mhsa_config(num_heads=8, key_dim=64, value_dim=64, feature_height=12),
    }
)


# ============================================================================
# Neural Network Building Blocks
# ============================================================================

class ConvBNAct(nn.Layer):
    """Convolution + Batch Normalization + Activation block.
    
    Standard building block combining convolution, batch norm, and optional activation.
    Used throughout the network for feature extraction.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            groups=1,
            act=None,
            norm_type="batchnorm2d",
    ):
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
        self.norm_type = norm_type.lower()
        if self.norm_type in ["batchnorm2d", "batchnorm", "bn"]:
            self.norm = BatchNorm2D(
                out_channels,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0)),
            )
        elif self.norm_type in ["layernorm", "ln"]:
            self.norm = LayerNorm(out_channels)
        else:
            raise ValueError(
                "Unsupported norm_type '{}'. Use 'batchnorm2d' or 'layernorm'.".format(
                    norm_type
                )
            )
        self.act = act
        self.stride = stride

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type in ["layernorm", "ln"]:
            x = x.transpose([0, 2, 3, 1])
            x = self.norm(x)
            x = x.transpose([0, 3, 1, 2])
        else:
            x = self.norm(x)
        if self.act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hardswish":
                x = F.hardswish(x)
        return x


class MNV4LayerScale(nn.Layer):
    """Layer Scale module for MobileNetV4.
    
    Applies learnable channel-wise scaling as introduced in CaiT.
    Improves training stability and convergence for deeper networks.
    Reference: https://arxiv.org/abs/2103.17239
    """

    def __init__(self, num_channels, init_values=1e-5):
        super().__init__()
        self.gamma = self.create_parameter(
            shape=[num_channels],
            default_initializer=nn.initializer.Constant(init_values),
            attr=ParamAttr(),
        )
        self.add_parameter("gamma", self.gamma)

    def forward(self, x):
        # Reshape gamma to [1, C, 1, 1] and ensure dtype consistency
        gamma = self.gamma.reshape([1, -1, 1, 1]).astype(x.dtype)
        return x * gamma


class MultiQueryAttentionLayerWithDownSampling(nn.Layer):
    """Multi-Query Attention with spatial downsampling.
    
    Efficient attention mechanism optimized for mobile deployment.
    Uses shared keys/values across multiple query heads and supports
    independent spatial downsampling for queries and key-value pairs.
    """

    def __init__(
            self,
            input_channels,
            num_heads,
            key_dim,
            value_dim,
            query_h_strides=1,
            query_w_strides=1,
            kv_strides=1,
            dw_kernel_size=3,
            dropout=0.0,
            norm_type="batchnorm2d",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides
        self.dw_kernel_size = dw_kernel_size
        self.norm_type = norm_type.lower()

        if self.norm_type not in ["batchnorm2d", "batchnorm", "bn", "layernorm", "ln"]:
            raise ValueError(
                "Unsupported norm_type '{}'. Use 'batchnorm2d' or 'layernorm'.".format(
                    norm_type
                )
            )

        # Query projection (multi-head)
        self.query_proj = Conv2D(
            in_channels=input_channels,
            out_channels=num_heads * key_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=False,
        )

        # Key and value projections with optional spatial downsampling
        self.key_dw_conv = None
        self.value_dw_conv = None
        self.key_dw_norm = None
        self.value_dw_norm = None

        if kv_strides > 1:
            # Add depthwise convolution for spatial downsampling
            self.key_dw_conv = Conv2D(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=dw_kernel_size,
                stride=kv_strides,
                padding=dw_kernel_size // 2,
                groups=input_channels,
                bias_attr=False,
            )
            self.value_dw_conv = Conv2D(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=dw_kernel_size,
                stride=kv_strides,
                padding=dw_kernel_size // 2,
                groups=input_channels,
                bias_attr=False,
            )
            if self.norm_type in ["layernorm", "ln"]:
                self.key_dw_norm = LayerNorm(input_channels)
                self.value_dw_norm = LayerNorm(input_channels)
            else:
                self.key_dw_norm = BatchNorm2D(input_channels)
                self.value_dw_norm = BatchNorm2D(input_channels)

        # 1x1 projection layers
        self.key_proj = Conv2D(
            in_channels=input_channels,
            out_channels=key_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=False,
        )

        self.value_proj = Conv2D(
            in_channels=input_channels,
            out_channels=value_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=False,
        )

        # Output projection to restore channel dimension
        self.output_proj = Conv2D(
            in_channels=num_heads * value_dim,
            out_channels=input_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=False,
        )

        self.dropout = nn.Dropout(dropout)

    def _apply_2d_norm(self, tensor, norm_layer):
        if norm_layer is None:
            return tensor
        if self.norm_type in ["layernorm", "ln"]:
            tensor = tensor.transpose([0, 2, 3, 1])
            tensor = norm_layer(tensor)
            tensor = tensor.transpose([0, 3, 1, 2])
            return tensor
        return norm_layer(tensor)

    def forward(self, x):
        B, C, H, W = x.shape

        # Query projection and reshape
        q = self.query_proj(x)  # [B, num_heads*key_dim, H, W]
        q = q.reshape([B, self.num_heads, self.key_dim, H * W])
        q = q.transpose([0, 1, 3, 2])  # [B, num_heads, H*W, key_dim]

        # Key projection and reshape
        key_input = x
        if self.key_dw_conv is not None:
            key_input = self.key_dw_conv(key_input)
            key_input = self._apply_2d_norm(key_input, self.key_dw_norm)
        k = self.key_proj(key_input)  # [B, key_dim, H', W']
        _, _, H_k, W_k = k.shape
        k = k.reshape([B, self.key_dim, H_k * W_k])
        k = k.transpose([0, 2, 1])  # [B, H'*W', key_dim]
        k = k.unsqueeze(1)  # [B, 1, H'*W', key_dim] for broadcasting across num_heads

        # Value projection and reshape
        value_input = x
        if self.value_dw_conv is not None:
            value_input = self.value_dw_conv(value_input)
            value_input = self._apply_2d_norm(value_input, self.value_dw_norm)
        v = self.value_proj(value_input)  # [B, value_dim, H', W']
        _, _, H_v, W_v = v.shape
        v = v.reshape([B, self.value_dim, H_v * W_v])
        v = v.transpose([0, 2, 1])  # [B, H'*W', value_dim]
        v = v.unsqueeze(1)  # [B, 1, H'*W', value_dim] for broadcasting across num_heads

        # Attention computation [B, num_heads, H*W, H'*W']
        attn = paddle.matmul(q, k, transpose_y=True)
        scale = paddle.to_tensor(self.key_dim ** 0.5, dtype=attn.dtype)
        attn = attn / scale
        attn = F.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        # Apply attention to values [B, num_heads, H*W, value_dim]
        out = paddle.matmul(attn, v)  # [B, num_heads, H*W, value_dim]

        # Reshape output [B, H*W, num_heads*value_dim]
        out = out.transpose([0, 2, 1, 3])
        out = out.reshape([B, H * W, self.num_heads * self.value_dim])

        # Reshape back to spatial [B, num_heads*value_dim, H, W]
        out = out.transpose([0, 2, 1])
        out = out.reshape([B, self.num_heads * self.value_dim, H, W])

        # Output projection
        out = self.output_proj(out)
        return out


class MultiHeadSelfAttentionBlock(nn.Layer):
    """Multi-Head Self-Attention block with normalization and residual connection.
    
    Combines batch normalization, multi-query attention, optional layer scaling,
    and residual connection for efficient feature interaction.
    """

    def __init__(
            self,
            input_channels,
            num_heads,
            key_dim,
            value_dim,
            query_h_strides=1,
            query_w_strides=1,
            kv_strides=1,
            use_layer_scale=False,
            use_multi_query=True,
            use_residual=True,
            norm_type="batchnorm2d",
    ):
        super().__init__()
        self.use_layer_scale = use_layer_scale
        self.use_multi_query = use_multi_query
        self.use_residual = use_residual
        self.norm_type = norm_type.lower()

        if self.norm_type in ["batchnorm2d", "batchnorm", "bn"]:
            self.input_norm = BatchNorm2D(input_channels)
        elif self.norm_type in ["layernorm", "ln"]:
            self.input_norm = LayerNorm(input_channels)
        else:
            raise ValueError(
                "Unsupported norm_type '{}'. Use 'batchnorm2d' or 'layernorm'.".format(
                    norm_type
                )
            )

        if use_multi_query:
            self.mqa = MultiQueryAttentionLayerWithDownSampling(
                input_channels,
                num_heads,
                key_dim,
                value_dim,
                query_h_strides,
                query_w_strides,
                kv_strides,
                norm_type=norm_type,
            )
        else:
            # Fallback to standard multi-head attention (not implemented)
            self.mqa = None

        if use_layer_scale:
            self.layer_scale = MNV4LayerScale(input_channels, init_values=1e-5)
        else:
            self.layer_scale = None

    def forward(self, x):
        shortcut = x
        if self.norm_type in ["layernorm", "ln"]:
            x = x.transpose([0, 2, 3, 1])
            x = self.input_norm(x)
            x = x.transpose([0, 3, 1, 2])
        else:
            x = self.input_norm(x)

        if self.mqa is not None:
            x = self.mqa(x)

        if self.layer_scale is not None:
            x = self.layer_scale(x)

        if self.use_residual:
            # Ensure same dtype for residual connection to avoid type promotion warning
            if shortcut.dtype != x.dtype:
                shortcut = shortcut.astype(x.dtype)
            x = x + shortcut

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
            act="relu",
            norm_type="batchnorm2d",
    ):
        super().__init__()
        self.start_dw_kernel_size = start_dw_kernel_size
        self.middle_dw_kernel_size = middle_dw_kernel_size
        self.use_layer_scale = use_layer_scale
        self.act = act
        self.stride = stride
        self.norm_type = norm_type

        if start_dw_kernel_size:
            start_stride = stride if not middle_dw_downsample else 1
            self.start_dw_conv = ConvBNAct(
                in_channels,
                in_channels,
                start_dw_kernel_size,
                stride=start_stride,
                groups=in_channels,
                act=None,
                norm_type=norm_type,
            )
        else:
            self.start_dw_conv = None

        expand_channels = make_divisible(in_channels * expand_ratio, 8)
        self.expand_conv = ConvBNAct(
            in_channels,
            expand_channels,
            1,
            act=act,
            norm_type=norm_type,
        )

        if middle_dw_kernel_size:
            middle_stride = stride if middle_dw_downsample else 1
            self.middle_dw_conv = ConvBNAct(
                expand_channels,
                expand_channels,
                middle_dw_kernel_size,
                stride=middle_stride,
                groups=expand_channels,
                act=act,
                norm_type=norm_type,
            )
        else:
            self.middle_dw_conv = None

        self.proj_conv = ConvBNAct(
            expand_channels,
            out_channels,
            1,
            act=None,
            norm_type=norm_type,
        )

        if use_layer_scale:
            self.layer_scale = MNV4LayerScale(out_channels, layer_scale_init_value)
        else:
            self.layer_scale = None

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
        if self.layer_scale is not None:
            x = self.layer_scale(x)
        if self.identity:
            # Ensure same dtype for residual connection to avoid type promotion warning
            if shortcut.dtype != x.dtype:
                shortcut = shortcut.astype(x.dtype)
            return x + shortcut
        return x


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
            norm_type="batchnorm2d",
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
        elif model_name == "hybrid_tiny":
            self.block_specs = block_specs_hybrid_tiny
        elif model_name == "hybrid_small":
            self.block_specs = block_specs_hybrid_small
        elif model_name == "hybrid_medium":
            self.block_specs = block_specs_hybrid_medium
        elif model_name == "hybrid_large":
            self.block_specs = block_specs_hybrid_large
        else:
            raise NotImplementedError(
                "mode[" + model_name + "_model] is not implemented!"
            )
        self.stride_overrides = stride_overrides or {}
        self.act = "hardswish"
        self.norm_type = norm_type
        if self.norm_type.lower() not in ["batchnorm2d", "batchnorm", "bn", "layernorm", "ln"]:
            raise ValueError(
                "Unsupported norm_type '{}'. Use 'batchnorm2d' or 'layernorm'.".format(
                    norm_type
                )
            )

        # Build feature extraction layers from block specifications
        layers = []
        current_channels = in_channels

        for block_idx, (block_type, *block_config) in enumerate(self.block_specs):
            stride = self.stride_overrides.get(block_idx)

            if block_type == "conv_bn":
                # Standard convolution block: (kernel_size, stride, out_channels)
                kernel_size, default_stride, out_channels = block_config
                if stride is None:
                    stride = default_stride
                layers.append(
                    ConvBNAct(current_channels, out_channels, kernel_size,
                              stride=stride, act=self.act, norm_type=self.norm_type)
                )
                current_channels = out_channels

            elif block_type == "uib":
                # Universal Inverted Bottleneck block
                # Format: (start_dw_kernel, middle_dw_kernel, middle_dw_downsample, 
                #          stride, out_channels, expand_ratio, use_layer_scale, [optional: mhsa_config])
                (start_dw_kernel, middle_dw_kernel, middle_dw_downsample,
                 default_stride, out_channels, expand_ratio, uib_use_layer_scale) = block_config[:7]
                mhsa_config = block_config[7] if len(block_config) > 7 else None

                if stride is None:
                    stride = default_stride

                layers.append(
                    UniversalInvertedBottleneck(
                        current_channels,
                        out_channels,
                        expand_ratio,
                        start_dw_kernel,
                        middle_dw_kernel,
                        stride,
                        middle_dw_downsample=middle_dw_downsample,
                        use_layer_scale=uib_use_layer_scale or use_layer_scale,
                        layer_scale_init_value=layer_scale_init_value,
                        act=self.act,
                        norm_type=self.norm_type,
                    )
                )
                current_channels = out_channels

                # Add MHSA block after UIB if configured
                if mhsa_config is not None:
                    (
                        num_heads, key_dim, value_dim, query_h_strides, query_w_strides,
                        kv_strides, mhsa_use_layer_scale, use_multi_query, use_residual
                    ) = mhsa_config
                    layers.append(
                        MultiHeadSelfAttentionBlock(
                            current_channels,
                            num_heads=num_heads,
                            key_dim=key_dim,
                            value_dim=value_dim,
                            query_h_strides=query_h_strides,
                            query_w_strides=query_w_strides,
                            kv_strides=kv_strides,
                            use_layer_scale=mhsa_use_layer_scale or use_layer_scale,
                            use_multi_query=use_multi_query,
                            use_residual=use_residual,
                            norm_type=self.norm_type,
                        )
                    )
            else:
                raise NotImplementedError("Unknown block type: {}".format(block_type))

        self.features = nn.Sequential(*layers)

        # Final pooling layer
        if last_pool_type == "avg":
            self.pool = nn.AvgPool2D(
                kernel_size=last_pool_kernel_size,
                stride=last_pool_kernel_size,
                padding=0,
            )
        else:
            self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        self.out_channels = current_channels

    def forward(self, inputs):
        x = self.features(inputs)
        x = self.pool(x)
        return x
