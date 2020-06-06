# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F
import copy
import math

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.structures.bounding_box import ObjectList
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.td_coder import TdCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import *


class FastRCNNLossComputation(object):
    def __init__(
        self,
        proposal_matcher,
        fg_bg_sampler,
        box_coder,
        td_coder,
        cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.td_coder = td_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target[0], proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)

        target = copy.deepcopy(target[1])
        #target = target[1].copy_with_fields(["labels", "left_box", "right_box"])
        matched_targets = target[matched_idxs.clamp(min=0)]
        return matched_targets, matched_idxs

    def prepare_targets(self, proposals, targets):
        results = []
        proposals_left = proposals[0]
        proposals_right = proposals[1]
        for proposals_per_image_left, proposals_per_image_right, targets_per_image in \
            zip(proposals_left, proposals_right, targets):

            proposals_union = get_union_boxes(proposals_per_image_left, proposals_per_image_right)
            targets_per_image_left = targets_per_image.get_field("left_box")
            targets_per_image_right = targets_per_image.get_field("right_box")
            targets_union = get_union_boxes(targets_per_image_left, targets_per_image_right)
            
            matched_targets, matched_idxs = self.match_targets_to_proposals(
                proposals_union, [targets_union, targets_per_image]
            )

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            box_regression_targets_per_image_left = self.box_coder.encode(
                matched_targets.get_field("left_box").bbox, proposals_per_image_left.bbox
            )
            proposals_per_image_left.add_field("regression_targets", box_regression_targets_per_image_left)
            proposals_per_image_left.add_field("labels", labels_per_image)

            box_regression_targets_per_image_right = self.box_coder.encode(
                matched_targets.get_field("right_box").bbox, proposals_per_image_right.bbox
            )
            proposals_per_image_right.add_field("regression_targets", box_regression_targets_per_image_right)
            proposals_per_image_right.add_field("labels", labels_per_image)

            #3d param
            dimension_regression_targets_per_image = \
                self.td_coder.dimentions_encode(matched_targets.get_field("dimensions"))
            rotation_label_per_image, rotation_regerssion_per_image = \
                self.td_coder.rotation_y_encode_v2(matched_targets.get_field("alpha"))

            #centers
            center_regerssion_targets_per_image_left = self.td_coder.centers_encode(
                matched_targets.get_field("left_centers"), proposals_per_image_left.bbox
            )
            proposals_per_image_left.add_field("center_regerssion_target", center_regerssion_targets_per_image_left)

            center_regerssion_targets_per_image_right = self.td_coder.centers_encode(
                matched_targets.get_field("right_centers"), proposals_per_image_right.bbox
            )
            proposals_per_image_right.add_field("center_regerssion_target", center_regerssion_targets_per_image_right)

            result = ObjectList()
            result.add_field('label', labels_per_image)
            result.add_field('proposals_left', proposals_per_image_left)
            result.add_field('proposals_right', proposals_per_image_right)
            result.add_field('dimension_regression_target', dimension_regression_targets_per_image)
            result.add_field('rotation_label', rotation_label_per_image)
            result.add_field('rotation_regerssion_target', rotation_regerssion_per_image)
            results.append(result)

        return results

    def subsample(self, proposals, targets):
        results = self.prepare_targets(proposals, targets)
        labels = [res.get_field('label') for res in results]
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            results_per_img = results[img_idx][img_sampled_inds]
            results[img_idx] = results_per_img

        return results

    def __call__(self, pred, targets):
        #2d box
        class_logits = pred['score']
        box_regression_left, box_regression_right = split_box(pred['bbox_reg'])
        proposal_left = [target.get_field("proposals_left") for target in targets]
        loss_classifier_left, loss_box_reg_left = self.cal_box_loss([class_logits], [box_regression_left], proposal_left)

        proposal_right = [target.get_field("proposals_right") for target in targets]
        loss_classifier_right, loss_box_reg_right = self.cal_box_loss([class_logits], [box_regression_right], proposal_right)

        loss_classifier = (loss_classifier_left + loss_classifier_right) * 0.5 * 0.2
        loss_box_reg = (loss_box_reg_left + loss_box_reg_right) * 0.5 * 0.2

        #2d centers
        center_regression_left, center_regression_right =  split_center(pred['center_reg'])
        loss_center_left = self.cal_center_loss([center_regression_left], proposal_left)
        loss_center_right = self.cal_center_loss([center_regression_right], proposal_right)
        loss_center = (loss_center_left + loss_center_right)*0.8

        #3d params
        hwl_regression = pred['hwl_reg']
        loss_dimention = self.cal_hwl_loss([hwl_regression], targets)

        rotation_logits = pred['alpha_logit']
        rotation_regerssion = pred['alpha_reg']
        rot_classification_loss, rot_regression_loss = self.rotation_loss([rotation_logits], [rotation_regerssion], targets)

        
        loss_roi = dict(
            loss_classifier=loss_classifier, 
            loss_box_reg=loss_box_reg,
            loss_center=loss_center,
            #loss_center_left = loss_center_left,
            #loss_center_right = loss_center_right,
            loss_dimention = loss_dimention,
            rot_classification_loss = rot_classification_loss,
            rot_regression_loss = rot_regression_loss
        )
        
        return loss_roi

    def rotation_loss(self, rot_logits, rot_regression, targets):
        rot_logits = cat(rot_logits, dim=0)
        rot_regression = cat(rot_regression, dim=0)
        rot_label_target = cat([tar.get_field("rotation_label") for tar in targets], dim=0)
        rot_regression_target = cat([tar.get_field("rotation_regerssion_target") for tar in targets], dim=0)

        device = rot_regression.device

        if (not hasattr(self, "labels_pos")) or (not hasattr(self, "sampled_pos_inds_subset")):
            raise RuntimeError("cal_box_loss needs to be called before")

        map_inds =  self.labels_pos[:, None]
        rot_classification_loss = F.cross_entropy(
            rot_logits[self.sampled_pos_inds_subset], rot_label_target[self.sampled_pos_inds_subset]
        ) * 0.2
        
        rot_regression_loss = smooth_l1_loss(
            rot_regression[self.sampled_pos_inds_subset, map_inds],
            rot_regression_target[self.sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )

        rot_regression_loss = rot_regression_loss / rot_regression_target.numel() * 0.1
        
        return rot_classification_loss, rot_regression_loss


    def cal_hwl_loss(self, hwl_regression, targets):
        hwl_regression = cat(hwl_regression, dim=0)
        hwl_regression_target = cat([tar.get_field("dimension_regression_target") for tar in targets], dim=0)
        device = hwl_regression.device

        if (not hasattr(self, "labels_pos")) or (not hasattr(self, "sampled_pos_inds_subset")):
            raise RuntimeError("cal_box_loss needs to be called before")
        map_inds =  3*self.labels_pos[:, None] + torch.tensor([0,1,2], device=device)
        hwl_loss = smooth_l1_loss(
            hwl_regression[self.sampled_pos_inds_subset[:, None], map_inds],
            hwl_regression_target[self.sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
        hwl_loss = hwl_loss / hwl_regression_target.numel() / 3.0 
        return hwl_loss


    def cal_center_loss(self, center_regression, proposals):
        center_regression = cat(center_regression, dim=0)
        center_regression_targets = cat(
            [proposal.get_field("center_regerssion_target") for proposal in proposals], dim=0
        )
        device = center_regression.device

        if (not hasattr(self, "labels_pos")) or (not hasattr(self, "sampled_pos_inds_subset")):
            raise RuntimeError("cal_box_loss needs to be called before")
        map_inds =  2*self.labels_pos[:, None] + torch.tensor([0,1], device=device)

        centers_loss = smooth_l1_loss(
            center_regression[self.sampled_pos_inds_subset[:, None], map_inds],
            center_regression_targets[self.sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
        centers_loss = centers_loss / center_regression_targets.numel() / 2.0
        return centers_loss


    def cal_box_loss(self, class_logits, box_regression, proposals):
        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        classification_loss = F.cross_entropy(class_logits, labels)

        self.sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        self.labels_pos = labels[self.sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * self.labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)

        box_loss = smooth_l1_loss(
            box_regression[self.sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[self.sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)
    td_coder = TdCoder(num_angle_bin=cfg.MODEL.ROI_BOX_HEAD.NUM_ROT_BIN)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        td_coder,
        cls_agnostic_bbox_reg
    )

    return loss_evaluator
