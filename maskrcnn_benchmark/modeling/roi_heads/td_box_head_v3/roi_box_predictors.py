# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
import torch
from torch import nn
import math
from torch.nn import functional as F
from torch.autograd import Variable


@registry.ROI_BOX_TD_V3_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred


@registry.ROI_BOX_TD_V3_PREDICTOR.register("FPNPredictor")
class FPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        '''
        this is used!!!!!!!!!!!!!!!!!!!!
        '''
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_angle_bin = cfg.MODEL.ROI_BOX_HEAD.NUM_ROT_BIN
        representation_size = in_channels

        self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.num_bbox_reg_classes = num_bbox_reg_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 6)

        self.hwl_pred = nn.Linear(representation_size, num_bbox_reg_classes * 3)
        self.alpha_logit = nn.Linear(representation_size, 2)
        self.alpha_pred = nn.Linear(representation_size, num_bbox_reg_classes * self.num_angle_bin)
        self.center_pred = nn.Linear(representation_size, num_bbox_reg_classes * 3)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)

        nn.init.normal_(self.hwl_pred.weight, mean=0, std=0.001)
        nn.init.normal_(self.alpha_logit.weight, mean=0, std=0.01)
        nn.init.normal_(self.alpha_pred.weight, mean=0, std=0.01)
        nn.init.normal_(self.center_pred.weight, mean=0, std=0.001)

        for l in [
            self.cls_score, 
            self.bbox_pred, 
            self.hwl_pred, 
            self.alpha_logit, 
            self.alpha_pred, 
            self.center_pred
        ]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)

        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        hwl_reg = self.hwl_pred(x)
        alpha_logit = self.alpha_logit(x)
        alpha_reg = self.alpha_pred(x)
        center_reg = self.center_pred(x)

        alpha_reg = alpha_reg.view(-1, self.num_bbox_reg_classes, self.num_angle_bin)
        rot_regression_prob = F.softmax(alpha_reg, -1)
        rot_regression = Variable(torch.FloatTensor(rot_regression_prob.size()[0], self.num_bbox_reg_classes).zero_()).cuda()
        for i in range(self.num_angle_bin):
            rot_regression += rot_regression_prob[:, :, i] * (math.pi / (self.num_angle_bin - 1)) * i
        #rot_regression = rot_regression.contiguous()

        res = {
            'score': scores,
            'bbox_reg': bbox_deltas,
            'hwl_reg': hwl_reg,
            'alpha_logit': alpha_logit,
            'alpha_reg': rot_regression,
            'center_reg': center_reg
        }

        return res


def make_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_TD_V3_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels)
