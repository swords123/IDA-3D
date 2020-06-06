# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList

from .inference import make_cost_volum_post_processor
from .loss import make_cost_volum_loss_evaluator
from .depth_cost_feature_extractors import make_depth_cost_feature_extractor

def keep_only_positive_boxes(boxes):
    boxes_left, boxes_right = boxes
    assert isinstance(boxes_left, (list, tuple)) and isinstance(boxes_right, (list, tuple))
    assert isinstance(boxes_left[0], BoxList) and isinstance(boxes_right[0], BoxList)
    assert boxes_left[0].has_field("labels") and boxes_right[0].has_field("labels")
    positive_boxes_left  = []
    positive_boxes_right= []
    for boxes_left_per_image, boxes_right_per_image in zip(boxes_left, boxes_right):
        labels = boxes_left_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes_left.append(boxes_left_per_image[inds])
        positive_boxes_right.append(boxes_right_per_image[inds])
    return positive_boxes_left, positive_boxes_right


class ROICostHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROICostHead, self).__init__()
        self.cfg = cfg.clone()

        self.post_processor = make_cost_volum_post_processor(cfg)
        self.loss_evaluator = make_cost_volum_loss_evaluator(cfg)
        self.depth_feature_extractors = make_depth_cost_feature_extractor(cfg,in_channels)

    def forward(self, features, proposals, targets, calib):
        if self.training:
            proposals_left, proposals_right = keep_only_positive_boxes(proposals)
            proposals_sum_num = 0
            for proposal_left_per_image, proposal_right_per_image in zip(proposals_left, proposals_right):
                proposals_sum_num = proposals_sum_num + len(proposals_left)
            if proposals_sum_num == 0:
                return features, proposals, dict(loss_cost=0.0)
        else:
            assert all([pro.is_equal() for pro in proposals])
            proposals_left = [pro.get_field("left_box") for pro in proposals]
            proposals_right = [pro.get_field("right_box") for pro in proposals]

        depth = self.depth_feature_extractors(features, [proposals_left, proposals_right], calib)
        
        if not self.training:
            result = self.post_processor(depth, proposals)
            return features, result, {}
        
        loss_cost = self.loss_evaluator(depth, [proposals_left, proposals_right], targets, calib)
        return features, proposals, dict(loss_cost_depth=loss_cost)


def build_cost_volum_v11_head(cfg, in_channels):
    return ROICostHead(cfg, in_channels)
