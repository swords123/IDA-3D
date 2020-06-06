# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from torch import nn
from torch.autograd import Variable

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.build_rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class MonoRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(MonoRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.uncert = Variable(torch.rand(4).cuda(), requires_grad=True)
        torch.nn.init.constant(self.uncert, -1)

    def forward(self, images_left, images_right, targets=None, calib=None):

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images_left = to_image_list(images_left)
        features = self.backbone(images_left.tensors)
        proposals, proposal_losses = self.rpn(images_left, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            proposal_losses["loss_objectness"] = proposal_losses["loss_objectness"] * torch.exp(-self.uncert[0]) + self.uncert[0] / 10
            proposal_losses["loss_rpn_box_reg"] = proposal_losses["loss_rpn_box_reg"] * torch.exp(-self.uncert[1]) + self.uncert[1] / 10
            detector_losses["loss_classifier"] = detector_losses["loss_classifier"] * torch.exp(-self.uncert[2]) + self.uncert[2] / 10
            detector_losses["loss_box_reg"] = detector_losses["loss_box_reg"] * torch.exp(-self.uncert[3]) + self.uncert[3] / 10

            print(self.uncert.data)
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
