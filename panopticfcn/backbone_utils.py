# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from typing import Dict
from torch import nn
from torch.nn import functional as F
import fvcore.nn.weight_init as weight_init

from detectron2.modeling.backbone import build_resnet_backbone
from detectron2.modeling.backbone.backbone import Backbone
from detectron2.modeling.backbone.fpn import FPN
from detectron2.layers import Conv2d, ShapeSpec, get_norm

class SemanticFPN(nn.Module):
    """
    A semantic segmentation head described in detail in the Panoptic Feature Pyramid Networks paper
    (https://arxiv.org/abs/1901.02446). It takes FPN features as input and merges information from
    all levels of the FPN into single output.
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.in_features      = cfg.MODEL.SEMANTIC_FPN.IN_FEATURES
        feature_strides       = {k: v.stride for k, v in input_shape.items()}
        feature_channels      = {k: v.channels for k, v in input_shape.items()}
        self.common_stride    = cfg.MODEL.SEMANTIC_FPN.COMMON_STRIDE
        conv_dims             = cfg.MODEL.SEMANTIC_FPN.CONVS_DIM
        norm                  = cfg.MODEL.SEMANTIC_FPN.NORM
        # fmt: on

        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            )
            for k in range(head_length):
                norm_module = get_norm(norm, conv_dims)
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=norm_module,
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])

    def forward(self, features):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])
        
        return x


class LastLevelP6P7fromP5(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    P5 feature.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.num_levels = 2
        self.in_feature = "p5"
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)
    
    def forward(self, p5):
        p6 = self.p6(p5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


def build_resnet_fpn_p5_backbone(cfg, input_shape: ShapeSpec):
    """
    Build ResNet-FPN backbone with P6 and P7 from P5 feature.

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_p6p7 = out_channels
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7fromP5(in_channels_p6p7, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


def build_backbone(cfg, input_shape=None):
    """
    Build a backbone for PanopticFCN.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone = build_resnet_fpn_p5_backbone(cfg, input_shape)
    assert isinstance(backbone, Backbone)
    return backbone


def build_semanticfpn(cfg, input_shape=None):
    return SemanticFPN(cfg, input_shape)