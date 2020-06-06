# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

class PostProcessor(nn.Module):
    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        num_class=2
    ):
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        self.num_class = num_class

    def forward(self, depth, proposals):
        for depth_per_img, proposal_per_img in zip(depth, proposals):
            depth_per_img = depth_per_img.unsqueeze(-1)
            disp_per_img = depth_per_img.new(depth_per_img.shape)
            position_z_per_img = torch.cat((depth_per_img, disp_per_img), -1)
            proposal_per_img.add_field("positions_z", position_z_per_img) 
            proposal_per_img.add_field("positions_z_depth", position_z_per_img) 
        return proposals

def make_cost_volum_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN
    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH

    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    num_class = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES

    postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        detections_per_img,
        num_class
    )
    return postprocessor
