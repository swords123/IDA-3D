# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.layers import FrozenBatchNorm2d

from .submodule import *

import numpy as np


def get_boxes_for_cost_volum(left_boxes, right_boxes, depth_bin_rate, calib_list):
    depth_max = 87

    max_depth = len(depth_bin_rate)
    proposals_left = []
    proposals_right = []
    depth_bin_list = []
    #box_num = 0
    for left_box, right_box, calib in zip(left_boxes, right_boxes, calib_list):
        mode = left_box.mode
        assert mode == 'xyxy'
        xmin = torch.min(left_box.bbox[:,0], right_box.bbox[:,0])
        ymin = torch.min(left_box.bbox[:,1], right_box.bbox[:,1])
        xmax = torch.max(left_box.bbox[:,2], right_box.bbox[:,2])
        ymax = torch.max(left_box.bbox[:,3], right_box.bbox[:,3])

        #disp_bin_per_image = disp_bin_rate * (xmax - xmin).view(-1,1)
        #disp_bin_per_image = calib['b'] * calib['fu'] / (depth_bin_rate * (xmax - xmin).view(-1,1)) / 2
        #disp_bin_per_image = calib['b'] * calib['fu'] / depth_bin_rate / 2 
        depth_bin_per_image_min = calib['b'] * calib['fu'] / ((xmax - xmin) * 0.9).view(-1,1)
        depth_bin_per_image = depth_max - (depth_max - depth_bin_per_image_min) * depth_bin_rate
        disp_bin_per_image = calib['b'] * calib['fu'] / depth_bin_per_image / 2
        depth_bin_list.append(depth_bin_per_image)

        bbox_shift_left_per_image = []
        bbox_shift_rigth_per_image = []
        for i in range(len(depth_bin_rate)):
            xmin_shift_left = xmin + disp_bin_per_image[:,i]
            xmax_shift_left = torch.clamp(xmax + disp_bin_per_image[:, i], max=left_box.size[0] - 1)
            bbox_shift_left = torch.stack((xmin_shift_left, ymin, xmax_shift_left, ymax), dim = 1)
            bbox_shift_left_per_image.append(BoxList(bbox_shift_left, left_box.size, mode="xyxy"))

            xmin_shift_right = torch.clamp(xmin - disp_bin_per_image[:, i], min=0)
            xmax_shift_right = xmax -disp_bin_per_image[:, i]
            bbox_shift_right = torch.stack((xmin_shift_right, ymin, xmax_shift_right, ymax), dim = 1)
            bbox_shift_rigth_per_image.append(BoxList(bbox_shift_right, right_box.size, mode="xyxy"))
        
        proposals_left.append(bbox_shift_left_per_image)
        proposals_right.append(bbox_shift_rigth_per_image)
    
    proposals_left = list(zip(*proposals_left))
    proposals_right = list(zip(*proposals_right))

    depth_bin = depth_bin_list[0]
    for i in range(1,len(depth_bin_list)):
        depth_bin = torch.cat((depth_bin,depth_bin_list[i]),0) 

    return proposals_left, proposals_right, depth_bin


class DepthCostFeatureExtractor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(DepthCostFeatureExtractor, self).__init__()

        self.depth_bin_rate = torch.tensor(cfg.MODEL.ROI_BOX_HEAD.DEPTH_BIN_RATE).cuda()
        self.max_depth = len(self.depth_bin_rate)

        #resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION * 2
        resolution = 16
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler
        self.reduced_channel = 32
        self.resolution = resolution

        #self.dim_reduce = nn.Conv2d(in_channels, 32, kernel_size=1, stride=1)
        self.dim_reduce = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, stride=1),
                                FrozenBatchNorm2d(64), nn.ReLU(inplace=True),
                                nn.Conv2d(64, 32, kernel_size=1, stride=1),
                                FrozenBatchNorm2d(32), nn.ReLU(inplace=True))
        #------------------------------lalala----------------------
        
        #self.avg_pool = nn.MaxPool2d(resolution,resolution)
        
        self.dres0 = nn.Sequential(convbn_3d(96, 64, 3, 1, 1),nn.ReLU(inplace=True),
                                     convbn_3d(64, 128, 3, 1, 1),nn.ReLU(inplace=True))

        self.max_pool1 = nn.MaxPool3d((1,2,2))

        self.dres1 = nn.Sequential(convbn_3d(128, 128, 3, 1, 1),nn.ReLU(inplace=True),
                                   convbn_3d(128, 128, 3, 1, 1)) 

        self.max_pool2 = nn.MaxPool3d((1,2,2))

        self.dres2 = nn.Sequential(convbn_3d(128, 64, 3, 1, 1),nn.ReLU(inplace=True),
                                    nn.Conv3d(64, 1, kernel_size=3, padding=1, stride=1,bias=False)) 
        
        self.avg_pool = nn.AvgPool2d(4,4)
        


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()       
    
    def depth_cost(self, cost, depth_bin, num_channels):
        cost = cost.contiguous()

        x_l_norm = torch.sqrt(torch.sum(cost[:, :num_channels,:,:,:]*cost[:, :num_channels,:,:,:],(1,3,4))) 
        x_r_norm = torch.sqrt(torch.sum(cost[:, num_channels:num_channels*2,:,:,:]*cost[:, num_channels:num_channels*2,:,:,:],(1,3,4)))
        x_cross  = torch.sum(cost[:, :num_channels,:,:,:]*cost[:, num_channels:num_channels*2,:,:,:],(1,3,4))/torch.clamp(x_l_norm*x_r_norm,min=0.01)
        x_cross = x_cross.unsqueeze(1).unsqueeze(3).unsqueeze(4)

        #np.save('raw.npy', cost.cpu().numpy())

        cost = self.dres0(cost)
        cost = self.max_pool1(cost)

        #np.save('f_1.npy', cost.cpu().numpy())

        cost = cost * x_cross
        
        cost = self.dres1(cost) + cost
        cost = self.max_pool2(cost)

        #np.save('f_2.npy', cost.cpu().numpy())

        cost = self.dres2(cost)

        cost_disp = torch.squeeze(cost, 1)
        cost_disp = self.avg_pool(cost_disp)
        cost_disp = cost_disp.squeeze(-1)
        cost_disp = cost_disp.squeeze(-1)

        disp_prob = F.softmax(cost_disp,-1)

        #np.save('a.npy', disp_prob.cpu().numpy())

        disp = Variable(torch.FloatTensor(disp_prob.size()[0]).zero_()).cuda()
        for i in range(self.max_depth):
            disp += disp_prob[:,i] * depth_bin[:,i]
        disp = disp.contiguous()
        
        return disp


    def forward(self, features, proposals, calib):
        proposals_left, proposals_right = proposals
        features_left, features_right = features

        proposals_shift_left, proposals_shift_right, depth_bin = get_boxes_for_cost_volum(proposals_left,proposals_right,self.depth_bin_rate, calib)
        
        features_left_reduce = []
        features_right_reduce = []
        for feature_left, fearure_right in zip(features_left, features_right):
            features_left_reduce.append(self.dim_reduce(feature_left))
            features_right_reduce.append(self.dim_reduce(fearure_right))
        features_left_reduce = tuple(features_left_reduce)
        features_right_reduce = tuple(features_right_reduce)

        num_channels = self.reduced_channel
        cost = Variable(torch.FloatTensor(depth_bin.size()[0], num_channels*3, self.max_depth, self.resolution, self.resolution).zero_()).cuda()
        idx = 0
        for proposals_s_l, proposals_s_r in zip(proposals_shift_left, proposals_shift_right):
            x_l = self.pooler(features_left_reduce, proposals_s_l)
            x_r = self.pooler(features_right_reduce, proposals_s_r)
            cost[:, :num_channels,idx,:,:] = x_l
            cost[:, num_channels : num_channels*2,idx,:,:] = x_r
            cost[:, num_channels*2 : num_channels*3,idx,:,:] = x_l-x_r
            idx += 1
        
        disp = self.depth_cost(cost, depth_bin, num_channels)
        disp = disp.split([len(box) for box in proposals_left], dim = 0)
        return disp


def make_depth_cost_feature_extractor(cfg, in_channels):
    func = DepthCostFeatureExtractor
    return func(cfg, in_channels)
