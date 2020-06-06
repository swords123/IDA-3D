# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.image_list import to_image_list

class BatchCollator(object):
    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images_left = to_image_list(transposed_batch[0], self.size_divisible)
        images_right = to_image_list(transposed_batch[1], self.size_divisible)
        targets = transposed_batch[2]
        calib = transposed_batch[3]
        img_ids = transposed_batch[4]
        return images_left, images_right, targets, calib, img_ids
