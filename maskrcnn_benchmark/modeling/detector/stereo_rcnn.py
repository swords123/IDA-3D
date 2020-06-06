# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from torch import nn
from torch.autograd import Variable

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.build_rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads

import numpy as np


class StereoRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(StereoRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        #self.uncert = Variable(torch.rand(8).cuda(), requires_grad=True)
        #torch.nn.init.constant(self.uncert, 0)

    def forward(self, images_left, images_right, targets=None, calib=None):

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images_left = to_image_list(images_left)
        images_right = to_image_list(images_right)

        features_left = self.backbone(images_left.tensors)
        features_right = self.backbone(images_right.tensors)
        
        images = [images_left, images_right]
        features = [features_left, features_right]

        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets, calib)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            #proposal_losses["loss_objectness"] = proposal_losses["loss_objectness"] * torch.exp(-self.uncert[0]) + self.uncert[0] / 5
            #proposal_losses["loss_rpn_box_reg"] = proposal_losses["loss_rpn_box_reg"] * torch.exp(-self.uncert[1]) + self.uncert[1] / 5
            #detector_losses["loss_classifier"] = detector_losses["loss_classifier"] * torch.exp(-self.uncert[2]) + self.uncert[2] / 5
            #detector_losses["loss_center"] = detector_losses["loss_center"] * torch.exp(-self.uncert[3]) + self.uncert[3] / 5
            #detector_losses["loss_dimention"] = detector_losses["loss_dimention"] * torch.exp(-self.uncert[4]) + self.uncert[4] / 5
            #detector_losses["rot_classification_loss"] = detector_losses["rot_classification_loss"] * torch.exp(-self.uncert[5]) + self.uncert[5] / 5
            #detector_losses["rot_regression_loss"] = detector_losses["rot_regression_loss"] * torch.exp(-self.uncert[6]) + self.uncert[6] / 5
            #detector_losses["loss_cost_depth"] = detector_losses["loss_cost_depth"] * torch.exp(-self.uncert[7]) + self.uncert[7] / 5
            #print(self.uncert.data)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
