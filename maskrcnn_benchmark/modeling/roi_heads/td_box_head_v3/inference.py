# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.bounding_box import ObjectList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms_stereo_td
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.td_coder import TdCoder

from maskrcnn_benchmark.modeling.utils import split_box
from maskrcnn_benchmark.modeling.utils import split_center


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=None,
        td_coder=None,
        cls_agnostic_bbox_reg=False,
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        if td_coder is None:
            td_coder = TdCoder(num_angle_bin=10)
        self.box_coder = box_coder
        self.td_coder = td_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def forward(self, predictions, boxes):
        box_regression_left, box_regression_right = split_box(predictions['bbox_reg'])
        class_logits = predictions['score']
        center_regression_left, center_regression_right = split_center(predictions['center_reg'])
        dimension_regression = predictions['hwl_reg']
        rotation_logits = predictions['alpha_logit']
        rotation_regression = predictions['alpha_reg']


        boxes_left, boxes_right = boxes
        class_prob = F.softmax(class_logits, -1)
        rotation_prob = F.softmax(rotation_logits, -1) 
        rotation_label = torch.argmax(rotation_prob, dim=1)
        #rotation_regression = torch.sigmoid(rotation_regression) * self.td_coder.rotation_bin

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes_left]
        boxes_per_image = [len(box) for box in boxes_left]

        concat_boxes_left = torch.cat([a.bbox for a in boxes_left], dim=0)
        concat_boxes_right = torch.cat([a.bbox for a in boxes_right], dim=0)

        proposals_left = self.box_coder.decode(box_regression_left.view(sum(boxes_per_image), -1), concat_boxes_left)
        proposals_right = self.box_coder.decode(box_regression_right.view(sum(boxes_per_image), -1), concat_boxes_right)
        centers_left = self.td_coder.centers_decode(center_regression_left, concat_boxes_left)
        centers_right = self.td_coder.centers_decode(center_regression_right, concat_boxes_right)
        dimensions = self.td_coder.dimentions_decode(dimension_regression)
        rotations = self.td_coder.rotation_y_decode_v2(rotation_regression, rotation_label)

        num_classes = class_prob.shape[1]

        class_prob = class_prob.split(boxes_per_image, dim=0)
        proposals_left = proposals_left.split(boxes_per_image, dim=0)
        proposals_right = proposals_right.split(boxes_per_image, dim=0)
        centers_left = centers_left.split(boxes_per_image, dim=0)
        centers_right = centers_right.split(boxes_per_image, dim=0)
        dimensions = dimensions.split(boxes_per_image, dim=0)
        rotations = rotations.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_img_left, boxes_per_img_right, \
            centers_per_img_left, centers_per_img_right, \
            dimensions_per_img, rotations_per_img, image_shape in zip(
            class_prob, proposals_left, proposals_right, \
            centers_left, centers_right, dimensions, rotations, image_shapes
        ):
            boxlist_left = self.prepare_boxlist(boxes_per_img_left, prob, image_shape)
            boxlist_right = self.prepare_boxlist(boxes_per_img_right, prob, image_shape)

            boxlist_left = boxlist_left.clip_to_image(remove_empty=False)
            boxlist_right = boxlist_right.clip_to_image(remove_empty=False)

            result = self.prepare_objectlist(
                [boxlist_left, boxlist_right], 
                [centers_per_img_left, centers_per_img_right],
                dimensions_per_img, rotations_per_img, prob
            )

            #boxlist_left, boxlist_right = self.filter_results([boxlist_left, boxlist_right], num_classes)
            result = self.filter_results(result, num_classes)

            results.append(result)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist
    
    def prepare_objectlist(self, boxlists, centers, dimensions, rotations, scores):
        boxlist_left, boxlist_right = boxlists
        centers_left, centers_right = centers
        centers_left = centers_left.reshape(-1, 2)
        centers_right = centers_right.reshape(-1, 2)
        dimensions = dimensions.reshape(-1,3)
        rotations = rotations.reshape(-1)
        scores = scores.reshape(-1)
        object_list = ObjectList()
        object_list.add_field("left_box", boxlist_left)
        object_list.add_field("right_box", boxlist_right)
        object_list.add_field("left_centers", centers_left)
        object_list.add_field("right_centers", centers_right)
        object_list.add_field("dimensions", dimensions)
        object_list.add_field("rotations", rotations)
        object_list.add_field("scores", scores)
        return object_list

    def filter_results(self, objectlist, num_classes):
        boxlist_left = objectlist.get_field("left_box")
        boxlist_right = objectlist.get_field("right_box")
        boxes_left = boxlist_left.bbox.reshape(-1, num_classes * 4)
        boxes_right = boxlist_right.bbox.reshape(-1, num_classes * 4)

        centers_left = objectlist.get_field("left_centers").reshape(-1, num_classes * 2)
        centers_right = objectlist.get_field("right_centers").reshape(-1, num_classes * 2)
        dimemsions = objectlist.get_field("dimensions").reshape(-1, num_classes * 3)
        rotations = objectlist.get_field("rotations").reshape(-1, num_classes)
        scores = objectlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result_box_left = []
        result_box_right = []
        result_center_left = []
        result_center_right = []
        result_dimensions = []
        result_rotations = []

        inds_all = scores > self.score_thresh

        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)

            scores_j = scores[inds, j]
            boxes_left_j = boxes_left[inds, j * 4 : (j + 1) * 4]
            boxes_right_j = boxes_right[inds, j * 4 : (j + 1) * 4]
            centers_left_j = centers_left[inds, j * 2 : (j + 1) * 2]
            centers_right_j = centers_right[inds, j * 2 : (j + 1) * 2]
            dimemsions_j = dimemsions[inds, j * 3 : (j + 1) * 3]
            rotations_j = rotations[inds, j]

            boxlist_left_for_class = BoxList(boxes_left_j, boxlist_left.size, mode="xyxy")
            boxlist_right_for_class = BoxList(boxes_right_j, boxlist_right.size, mode="xyxy")

            boxlist_left_for_class.add_field("scores", scores_j)
            boxlist_right_for_class.add_field("scores", scores_j)

            keep, mode = boxlist_nms_stereo_td(boxlist_left_for_class, boxlist_right_for_class, self.nms)
            boxlist_left_for_class = boxlist_left_for_class[keep].convert(mode)
            boxlist_right_for_class = boxlist_right_for_class[keep].convert(mode)
            centers_left_for_class = centers_left_j[keep]
            centers_right_for_class = centers_right_j[keep]
            dimemsions_for_class = dimemsions_j[keep]
            rotations_for_class = rotations_j[keep]

            num_labels = len(boxlist_left_for_class)
            labels = torch.full((num_labels,), j, dtype=torch.int64, device=device)

            boxlist_left_for_class.add_field("labels", labels)
            boxlist_right_for_class.add_field("labels", labels)

            result_box_left.append(boxlist_left_for_class)
            result_box_right.append(boxlist_right_for_class)
            result_center_left.append(centers_left_for_class)
            result_center_right.append(centers_right_for_class)
            result_dimensions.append(dimemsions_for_class)
            result_rotations.append(rotations_for_class)
            

        result_box_left = cat_boxlist(result_box_left)
        result_box_right = cat_boxlist(result_box_right)
        result_center_left = torch.cat(result_center_left)
        result_center_right = torch.cat(result_center_right)
        result_dimensions = torch.cat(result_dimensions)
        result_rotations = torch.cat(result_rotations)
        
        number_of_detections = len(result_box_left)

        result = ObjectList()
        result.add_field("left_box", result_box_left)
        result.add_field("right_box", result_box_right)
        result.add_field("left_centers", result_center_left)
        result.add_field("right_centers", result_center_right)
        result.add_field("dimensions", result_dimensions)
        result.add_field("rotations", result_rotations)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result_box_left.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]

        return result


def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)
    td_coder = TdCoder(num_angle_bin=cfg.MODEL.ROI_BOX_HEAD.NUM_ROT_BIN)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        detections_per_img,
        box_coder,
        td_coder,
        cls_agnostic_bbox_reg,
    )
    return postprocessor
