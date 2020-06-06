# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.bounding_box import BoxList

def get_union_boxes(left_box, right_box):
    mode = left_box.mode
    assert mode == 'xyxy'
    xmin = torch.min(left_box.bbox[:,0], right_box.bbox[:,0])
    ymin = torch.min(left_box.bbox[:,1], right_box.bbox[:,1])
    xmax = torch.max(left_box.bbox[:,2], right_box.bbox[:,2])
    ymax = torch.max(left_box.bbox[:,3], right_box.bbox[:,3])
    new_box = torch.stack((xmin,ymin,xmax,ymax) ,dim = 1)

    union_box = BoxList(new_box, left_box.size, mode='xyxy')
    return union_box

class CostLossComputation(object):
    def __init__(self, proposal_matcher):
        self.proposal_matcher = proposal_matcher

    def match_targets_to_proposals(self, proposal, target, object3d):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)

        matched_object3d = object3d[matched_idxs.clamp(min=0)]

        return matched_object3d, matched_idxs

    def prepare_targets(self, proposals, targets):
        disp_target = []
        position_target = []

        proposals_left = proposals[0]
        proposals_right = proposals[1]

        for proposals_per_image_left, proposals_per_image_right, targets_per_imgae in \
            zip(proposals_left, proposals_right, targets):

            proposals_union = get_union_boxes(proposals_per_image_left, proposals_per_image_right)
            targets_per_image_left = targets_per_imgae.get_field("left_box")
            targets_per_image_right = targets_per_imgae.get_field("right_box")
            targets_union = get_union_boxes(targets_per_image_left, targets_per_image_right)

            matched_object3d, matched_idxs = self.match_targets_to_proposals(
                proposals_union, targets_union, targets_per_imgae
            )

            disp_target.append(matched_object3d.get_field("positions_z")[:,0])  

        return disp_target

    def __call__(self, disp, proposals, targets, calib):

        disp_target = self.prepare_targets(proposals, targets)
        if len(disp_target) == 0:
            return 0.0

        disp_target = cat(disp_target, dim=0)
        disp = cat(disp, dim=0)

        cost_loss = smooth_l1_loss(
            disp,
            disp_target,
            size_average=False,
            beta=1,
        )
        cost_loss = cost_loss / disp_target.numel() * 0.2

        return cost_loss


def make_cost_volum_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    loss_evaluator = CostLossComputation(matcher)

    return loss_evaluator
