# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator

from maskrcnn_benchmark.structures.bounding_box import ObjectList
from maskrcnn_benchmark.modeling.utils import split_box


class ROIBoxHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        if self.training:
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)
                proposals_left = [pro.get_field("proposals_left") for pro in proposals]
                proposals_right = [pro.get_field("proposals_right") for pro in proposals]
        else:
            proposals_left, proposals_right = proposals

        x = self.feature_extractor(features, [proposals_left, proposals_right])
        pred = self.predictor(x)

        if not self.training:
            results = self.post_processor(pred, proposals)
            return x, results, {}

        loss_roi = self.loss_evaluator(pred, proposals)

        return (
            x,
            [proposals_left, proposals_right],
            loss_roi
        )


def build_td_box_head_v3(cfg, in_channels):
    return ROIBoxHead(cfg, in_channels)
