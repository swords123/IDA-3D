# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from maskrcnn_benchmark.modeling.rpn.mono_rpn.mono_rpn import build_mono_rpn
from maskrcnn_benchmark.modeling.rpn.stereo_rpn_v1.stereo_rpn_v1 import build_stereo_rpn_v1


def build_rpn(cfg, in_channels):
    if cfg.MODEL.RPN_VERSION == 0:
        return build_mono_rpn(cfg, in_channels)
    elif cfg.MODEL.RPN_VERSION == 1:
        return build_stereo_rpn_v1(cfg, in_channels)
