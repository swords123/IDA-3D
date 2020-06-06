# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Miscellaneous utility functions
"""

import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList

def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

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

def split_box(box_regression):
    N,C = box_regression.size()
    box_regression = box_regression.view(N,-1,6)
    box_regression_left = box_regression[:,:,[0,1,2,3]]
    box_regression_right = box_regression[:,:,[4,1,5,3]]
    box_regression_left = box_regression_left.view(N,-1)
    box_regression_right = box_regression_right.view(N,-1)
    return box_regression_left, box_regression_right

def split_center(center_regression):
    N,C = center_regression.size()
    center_regression = center_regression.view(N,-1,3)
    center_regression_left = center_regression[:,:,[0,2]]
    center_regression_right = center_regression[:,:,[1,2]]
    center_regression_left = center_regression_left.view(N,-1)
    center_regression_right = center_regression_right.view(N,-1)
    return center_regression_left, center_regression_right