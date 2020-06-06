# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py)
from __future__ import division

import os
from collections import defaultdict
import numpy as np
import math
import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

def do_kitti_evaluation(dataset, predictions, output_folder, logger):
    if predictions[0].has_field("left_box"):
        logger.info("2d left:")
        do_kitti_evaluation_2d(dataset, predictions, output_folder, logger, "left_box")
    if predictions[0].has_field("right_box"):
        logger.info("2d right:")
        do_kitti_evaluation_2d(dataset, predictions, output_folder, logger, "right_box")
    if predictions[0].has_other_field():
        do_kitti_evaluation_3d(dataset, predictions, output_folder, logger)

def do_kitti_evaluation_3d(dataset, predictions, output_folder, logger):
    gt_lists = []
    calib_lists = []
    img_info_lists = []
    for image_id, _ in enumerate(predictions):
        img_info_lists.append(dataset.get_img_info(image_id))
        gt_lists.append(dataset.get_groundtruth(image_id))
        calib_lists.append(dataset.preprocess_calib(image_id))
    
    gt_index_list = get_gt_index(gt_lists, predictions)
    
    '''
    if predictions[0].has_field("positions_z"):
        #convert disp to depth
        #predictions = depth2disp(predictions, calib_lists)
        predictions = disp2depth(predictions, calib_lists)
        result = cal_3d_error(gt_lists, predictions, gt_index_list, "positions_z")
        show_and_save_result(logger, output_folder, result, "positions_z")
        result_block = cal_depth_error(gt_lists, predictions, gt_index_list, "positions_z")
        show_block(result_block[1])
    '''

    if predictions[0].has_field("positions_z_1"):
        #convert disp to depth
        print('positions_z_1')
        predictions = rename_field(predictions, "positions_z_1")
        predictions = depth2disp(predictions, calib_lists)
        #predictions = disp2depth(predictions, calib_lists)
        result = cal_3d_error(gt_lists, predictions, gt_index_list, "positions_z")
        show_and_save_result(logger, output_folder, result, "positions_z")
        result_block = cal_depth_error(gt_lists, predictions, gt_index_list, "positions_z")
        show_block(result_block[1])
    
    if predictions[0].has_field("positions_z_2"):
        #convert disp to depth
        print('positions_z_2')
        predictions = rename_field(predictions, "positions_z_2")
        predictions = depth2disp(predictions, calib_lists)
        #predictions = disp2depth(predictions, calib_lists)
        result = cal_3d_error(gt_lists, predictions, gt_index_list, "positions_z")
        show_and_save_result(logger, output_folder, result, "positions_z")
        result_block = cal_depth_error(gt_lists, predictions, gt_index_list, "positions_z")
        show_block(result_block[1])

    if predictions[0].has_field("positions_z_depth"):
        #convert disp to depth
        print('positions_z_depth')
        predictions = rename_field(predictions, "positions_z_depth")
        predictions = depth2disp(predictions, calib_lists)
        #predictions = disp2depth(predictions, calib_lists)
        result = cal_3d_error(gt_lists, predictions, gt_index_list, "positions_z")
        show_and_save_result(logger, output_folder, result, "positions_z")
        result_block = cal_depth_error(gt_lists, predictions, gt_index_list, "positions_z")
        show_block(result_block[1])
    
    
    if predictions[0].has_field("left_centers"):
        result = cal_3d_error(gt_lists, predictions, gt_index_list, "left_centers")
        show_and_save_result(logger, output_folder, result, "left_centers")

        
        predictions = cal_xy(predictions, calib_lists)
        result = cal_3d_error(gt_lists, predictions, gt_index_list, "positions_xy")
        show_and_save_result(logger, output_folder, result, "positions_xy")
        

    if predictions[0].has_field("right_centers"):
        result = cal_3d_error(gt_lists, predictions, gt_index_list, "right_centers")
        show_and_save_result(logger, output_folder, result, "right_centers")

    if predictions[0].has_field("dimensions"):
        result = cal_3d_error(gt_lists, predictions, gt_index_list, "dimensions")
        show_and_save_result(logger, output_folder, result, "dimensions")
    
    if predictions[0].has_field("rotations"):
        predictions = convert_rotation(predictions, calib_lists)
        result = cal_3d_error(gt_lists, predictions, gt_index_list, "alpha")
        show_and_save_result(logger, output_folder, result, "alpha")
        result = cal_3d_error(gt_lists, predictions, gt_index_list, "beta")
        show_and_save_result(logger, output_folder, result, "beta")

def mean_z_corners(predictions, calib_lists):
    for pred, calib in zip(predictions, calib_lists):
        positions_z = pred.get_field("positions_z")
        zcorners = pred.get_field("z_corners")
        positions_z[:,0] = (positions_z[:,0] + torch.sum(zcorners, -1)) / 9
        positions_z[:,1] = calib["b"] * calib["fu"] / positions_z[:,0]
    return predictions 

def show_block(blocks):
    i = 0
    for block in blocks:
        print(str(i) + "-" + str(i + 10) + ": " + str(block[0][0]) + ', ' + \
            str(block[0][1]) + '      ' + str(block[1][0]) + ', ' + str(block[1][1]))
        i = i + 10

def merge_depth_and_disp(predictions, calib_lists):
    for pred, calib in zip(predictions, calib_lists):
        pred_depth = pred.get_field("positions_z_depth")
        pred_disp = pred.get_field("positions_z_disp")
        pred_depth[:,1] = calib["b"] * calib["fu"] / pred_depth[:,0]
        pred_disp[:,0] = calib["b"] * calib["fu"] / pred_disp[:,1]
        pred_z = [pos_dep if pos_dep[0] > 15 else pos_dis for pos_dep, pos_dis in zip(pred_depth, pred_disp)]
        if len(pred_z):
            pred_z = torch.stack(pred_z)
        else:
            pred_z = torch.tensor([]).view(-1,2)
        pred.add_field("positions_z", pred_z)
    return predictions 

def rename_field(predictions, filed_src):
    for pred in predictions:
        pred.add_field("positions_z",pred.get_field(filed_src))
    return predictions

def show_and_save_result(logger, output_folder, result, field):
    if isinstance(result, np.ndarray):
        result = [str(x) for x in result]
    elif isinstance(result, np.float32):
        result = [str(result)]
    result_str = field + " : " + ', '.join(result)
    logger.info(result_str)
    if output_folder:
        with open(os.path.join(output_folder, "result.txt"), "w+") as fid:
            fid.write(result_str)

def cal_xy(predictions, calib_lists):
    for pred, calib in zip (predictions, calib_lists):
        x_left = pred.get_field("positions_z")[:,0] * (pred.get_field("left_centers")[:,0] - calib["cu"]) / calib["fu"]
        y_left = pred.get_field("positions_z")[:,0] * (pred.get_field("left_centers")[:,1] - calib["cv"]) / calib["fv"]
        pred.add_field("positions_xy", torch.stack([x_left, y_left], dim=1))
    return predictions

def convert_rotation(predictions, calib_lists):
    for pred, calib in zip (predictions, calib_lists):
        pred.add_field("alpha", pred.get_field("rotations"))
        beta = pred.get_field("alpha") + torch.atan((pred.get_field("positions_xy")[:,0] - calib["bx2"]) / pred.get_field("positions_z")[:,0])
        pred.add_field("beta", beta)
    return predictions


def disp2depth(predictions, calib_lists):
    for pred, calib in zip(predictions, calib_lists):
        positions_z = pred.get_field("positions_z")
        positions_z[:,0] = calib["b"] * calib["fu"] / positions_z[:,1]
    return predictions    

def depth2disp(predictions, calib_lists):
    for pred, calib in zip(predictions, calib_lists):
        positions_z = pred.get_field("positions_z")
        positions_z[:,1] = calib["b"] * calib["fu"] / positions_z[:,0]
    return predictions   


def cal_3d_error(gt_lists, pred_lists, gt_index_list, field):
    distance = []
    for gt_list, pred_list, gt_index_per_image in zip(gt_lists, pred_lists, gt_index_list):
        pred_field = pred_list.get_field(field).numpy()
        gt_field = gt_list.get_field(field).numpy()

        pred_label = pred_list.get_field("left_box").get_field("labels").numpy()
        gt_label = gt_list.get_field("labels").numpy()
        
        for l in gt_index_per_image:
            pred_mask_l = pred_label == l
            pred_field_l = pred_field[pred_mask_l]
            pred_order = gt_index_per_image[l][0]
            pred_field_l = pred_field_l[pred_order]

            gt_mask_l = gt_label == l
            gt_field_l = gt_field[gt_mask_l]

            gt_index = gt_index_per_image[l][1]

            for idx in range(len(gt_index)):
                if gt_index[idx]>=0:
                    error_per_object = abs(pred_field_l[idx]-gt_field_l[gt_index[idx]])
                    distance.append(error_per_object)
    distance = np.array(distance)
    distance =  np.mean(distance,0) 
    return distance

def cal_depth_error(gt_lists, pred_lists, gt_index_list, field):
    distance = []
    distance_1 = []
    distance_2 = []
    distance_3 = []
    distance_4 = []
    distance_5 = []
    distance_6 = []
    distance_7 = []
    distance_8 = []
    for gt_list, pred_list, gt_index_per_image in zip(gt_lists, pred_lists, gt_index_list):
        pred_field = pred_list.get_field(field).numpy()
        gt_field = gt_list.get_field(field).numpy()

        pred_label = pred_list.get_field("left_box").get_field("labels").numpy()
        gt_label = gt_list.get_field("labels").numpy()
        
        for l in gt_index_per_image:
            pred_mask_l = pred_label == l
            pred_field_l = pred_field[pred_mask_l]
            pred_order = gt_index_per_image[l][0]
            pred_field_l = pred_field_l[pred_order]

            gt_mask_l = gt_label == l
            gt_field_l = gt_field[gt_mask_l]

            gt_index = gt_index_per_image[l][1]

            for idx in range(len(gt_index)):
                if gt_index[idx]>=0:
                    if gt_field_l[gt_index[idx],0] <= 10:
                        error_per_object = abs(pred_field_l[idx]-gt_field_l[gt_index[idx]])
                        distance_1.append(error_per_object)
                        distance.append(error_per_object)
                    if gt_field_l[gt_index[idx],0] > 10 and gt_field_l[gt_index[idx],0] <= 20:
                        error_per_object = abs(pred_field_l[idx]-gt_field_l[gt_index[idx]])
                        distance_2.append(error_per_object)
                        distance.append(error_per_object)
                    if gt_field_l[gt_index[idx],0] > 20 and gt_field_l[gt_index[idx],0] <= 30:
                        error_per_object = abs(pred_field_l[idx]-gt_field_l[gt_index[idx]])
                        distance_3.append(error_per_object)
                        distance.append(error_per_object)
                    if gt_field_l[gt_index[idx],0] > 30 and gt_field_l[gt_index[idx],0] <= 40:
                        error_per_object = abs(pred_field_l[idx]-gt_field_l[gt_index[idx]])
                        distance_4.append(error_per_object)
                        distance.append(error_per_object)
                    if gt_field_l[gt_index[idx],0] > 40 and gt_field_l[gt_index[idx],0] <= 50:
                        error_per_object = abs(pred_field_l[idx]-gt_field_l[gt_index[idx]])
                        distance_5.append(error_per_object)
                        distance.append(error_per_object)
                    if gt_field_l[gt_index[idx],0] > 50 and gt_field_l[gt_index[idx],0] <= 60:
                        error_per_object = abs(pred_field_l[idx]-gt_field_l[gt_index[idx]])
                        distance_6.append(error_per_object)
                        distance.append(error_per_object)
                    if gt_field_l[gt_index[idx],0] > 60 and gt_field_l[gt_index[idx],0] <= 70:
                        error_per_object = abs(pred_field_l[idx]-gt_field_l[gt_index[idx]])
                        distance_7.append(error_per_object)
                        distance.append(error_per_object)
                    if gt_field_l[gt_index[idx],0] > 70 and gt_field_l[gt_index[idx],0] <= 80:
                        error_per_object = abs(pred_field_l[idx]-gt_field_l[gt_index[idx]])
                        distance_8.append(error_per_object)
                        distance.append(error_per_object)                    
                    #error_per_object = abs(pred_field_l[idx]-gt_field_l[gt_index[idx]])
                    #distance.append(error_per_object)
    distance = np.array(distance)
    distance =  [np.mean(distance,0), np.var(distance,0)] 

    distance_1 = np.array(distance_1)
    distance_1 =  [np.mean(distance_1,0), np.var(distance_1,0)] 

    distance_2 = np.array(distance_2)
    distance_2 =  [np.mean(distance_2,0), np.var(distance_2,0)]

    distance_3 = np.array(distance_3)
    distance_3 =  [np.mean(distance_3,0), np.var(distance_3,0)]

    distance_4 = np.array(distance_4)
    distance_4 =  [np.mean(distance_4,0), np.var(distance_4,0)] 

    distance_5 = np.array(distance_5)
    distance_5 =  [np.mean(distance_5,0), np.var(distance_5,0)] 

    distance_6 = np.array(distance_6)
    distance_6 =  [np.mean(distance_6,0), np.var(distance_6,0)] 

    distance_7 = np.array(distance_7)
    distance_7 =  [np.mean(distance_7,0), np.var(distance_7,0)] 

    distance_8 = np.array(distance_8)
    distance_8 =  [np.mean(distance_8,0), np.var(distance_8,0)] 
    return distance, [distance_1, distance_2, distance_3, distance_4, distance_5, distance_6, distance_7, distance_8]

def get_gt_index(gt_lists, predictions, iou_thresh=0.5):
    score = defaultdict(list)
    match = defaultdict(list)

    assert len(gt_lists) == len(predictions), "Length of gt and pred lists need to be same."
    gt_index_list = []
    for gt_list, pred_list in zip(gt_lists, predictions):
        pred_bbox = pred_list.get_field("left_box").bbox.numpy()
        pred_label = pred_list.get_field("left_box").get_field("labels").numpy()
        pred_score = pred_list.get_field("left_box").get_field("scores").numpy()

        gt_bbox = gt_list.get_field("left_box").bbox.numpy()
        gt_label = gt_list.get_field("labels").numpy()
    
        gt_index_per_image = dict()
        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]

            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]

            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0 or len(gt_bbox_l) == 0:
                continue
            
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            iou = boxlist_iou(
                BoxList(pred_bbox_l, gt_list.get_field("left_box").size),
                BoxList(gt_bbox_l, gt_list.get_field("left_box").size),
            ).numpy()
            gt_index = iou.argmax(axis=1)
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou
            gt_index_per_image[l] = [order, gt_index]
        gt_index_list.append(gt_index_per_image)

    return gt_index_list


def do_kitti_evaluation_2d(dataset, predictions, output_folder, logger, field):
    pred_boxlists = []
    gt_boxlists = []
    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]

        prediction = prediction.get_field(field)
        prediction = prediction.resize((image_width, image_height))
        pred_boxlists.append(prediction)

        gt_boxlist = dataset.get_groundtruth(image_id).get_field(field)
        gt_boxlists.append(gt_boxlist)
    result = eval_detection_voc(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        iou_thresh=0.5,
        use_07_metric=True,
    )
    result_str = "mAP: {:.4f}".format(result["map"])
    logger.info(result_str)
    if output_folder:
        with open(os.path.join(output_folder, "result.txt"), "w+") as fid:
            fid.write(result_str)
    return result


def eval_detection_voc(pred_boxlists, gt_boxlists, iou_thresh=0.5, use_07_metric=False):
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."
    prec, rec = calc_detection_voc_prec_rec(
        pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh
    )
    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)
    return {"ap": ap, "map": np.nanmean(ap)}


def calc_detection_voc_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
   """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        pred_bbox = pred_boxlist.bbox.numpy()
        pred_label = pred_boxlist.get_field("labels").numpy()
        pred_score = pred_boxlist.get_field("scores").numpy()
        gt_bbox = gt_boxlist.bbox.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        gt_difficult = gt_boxlist.get_field("difficult").numpy()

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            iou = boxlist_iou(
                BoxList(pred_bbox_l, gt_boxlist.size),
                BoxList(gt_bbox_l, gt_boxlist.size),
            ).numpy()
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec


def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap
