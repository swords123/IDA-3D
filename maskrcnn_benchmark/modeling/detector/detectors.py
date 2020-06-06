# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .mono_rcnn import MonoRCNN
from .stereo_rcnn import StereoRCNN

_DETECTION_META_ARCHITECTURES = {"MonoRCNN": MonoRCNN, "StereoRCNN": StereoRCNN}

def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
