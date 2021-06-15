import os
import numpy as np
import cv2
import math as m
import matplotlib.pyplot as plt
from kitti_utils import *
import scipy.misc
import matplotlib.patches as patches
import write_xml
import copy

data_path = '/home/pwl/Work/unziped_data/kitti_object/training/'
CLASS = ['Car']

#label_ph_path = os.path.join(data_path, 'label_stereo_official')
#label_ph_path = './'

image_fliped_path = os.path.join(data_path, 'image_fliped')
image_2_fliped_path = os.path.join(image_fliped_path, 'image_2')
image_3_fliped_path = os.path.join(image_fliped_path, 'image_3')
label_fliped_path = os.path.join(image_fliped_path, 'label_3d')

if not os.path.exists(image_2_fliped_path):
    os.makedirs(image_2_fliped_path)

if not os.path.exists(image_3_fliped_path):
    os.makedirs(image_3_fliped_path)

if not os.path.exists(label_fliped_path):
    os.makedirs(label_fliped_path)

def load_img_from_index(index, pose = 'left'):
    if pose == 'left':
        img_file = os.path.join(data_path, 'image_2', index + '.png')
    elif pose == 'right':
        img_file = os.path.join(data_path, 'image_3', index + '.png')
    else:
        print 'ERROR'
        exit(0)
    img = cv2.imread(img_file)
    #img = img[:,:,(2,1,0)]
    return img

def write_anno(objects_origin, file_name, im_shape):
    objects = []
    objects_origin = remove_occluded_keypoints(objects_origin, left=True)
    objects_origin = remove_occluded_keypoints(objects_origin, left=False)

    for i in range(len(objects_origin)):
        if objects_origin[i].truncate < 0.98 and objects_origin[i].occlusion < 3 \
           and (objects_origin[i].boxes[0].box[3] - objects_origin[i].boxes[0].box[1]) > 10 \
           and objects_origin[i].cls in CLASS \
           and (objects_origin[i].boxes[0].visible_right - objects_origin[i].boxes[0].visible_left) > 3 \
           and (objects_origin[i].boxes[1].visible_right - objects_origin[i].boxes[1].visible_left) > 3: 
            objects.append(objects_origin[i])

    
    #if len(objects) == 0:
    #    return

    box = []
    for i in range(len(objects)):
        if not objects[i].cls in CLASS:
            continue
        
        pc = []
        for j in range(8):
            pc_str = '%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f'%(
                objects[i].eight_points_in_3d[j][0],
                objects[i].eight_points_in_3d[j][1],
                objects[i].eight_points_in_3d[j][2],
                objects[i].eight_points_in_left[j][0],
                objects[i].eight_points_in_left[j][1],
                objects[i].eight_points_in_right[j][0],
                objects[i].eight_points_in_right[j][1]
            )
            pc.append(pc_str)

        box.append({
            "left_box": objects[i].boxes[0].box, 
            "right_box": objects[i].boxes[1].box,
            "positions": objects[i].pos,
            "dimensions": objects[i].dim,
            "truncated": objects[i].truncate,
            "occluded": objects[i].occlusion,
            "alpha": objects[i].alpha,
            "rotation": objects[i].orientation,
            "disp": objects[i].center_in_2d_left[0] - objects[i].center_in_2d_right[0],
            "center_left": objects[i].center_in_2d_left,
            "center_right": objects[i].center_in_2d_right,
            "point_coner": pc
            })

    write_xml.write_xml(os.path.join(label_fliped_path,file_name + '.xml'), file_name, im_shape, box)

def flip_anno(objects, calib, im_shape):
    objects_fliped = copy.deepcopy(objects)
    for i in range(len(objects_fliped)):
        box_left_temp = objects[i].boxes[0].box
        box_right_temp = objects[i].boxes[1].box
        objects_fliped[i].boxes[0].box[0] = im_shape[1] - 1 - box_right_temp[2]
        objects_fliped[i].boxes[0].box[2] = im_shape[1] - 1 - box_right_temp[0]

        objects_fliped[i].boxes[1].box[0] = im_shape[1] - 1 - box_left_temp[2]
        objects_fliped[i].boxes[1].box[2] = im_shape[1] - 1 - box_left_temp[0]

        center_temp_left = objects[i].center_in_2d_left
        center_temp_right = objects[i].center_in_2d_right
        objects_fliped[i].center_in_2d_left[0] = im_shape[1] - 1 - center_temp_right[0]
        objects_fliped[i].center_in_2d_right[0] = im_shape[1] - 1 - center_temp_left[0]

        objects_fliped[i].orientation = m.pi - objects_fliped[i].orientation
        if objects_fliped[i].orientation >= m.pi:
            objects_fliped[i].orientation = objects_fliped[i].orientation - 2 * m.pi
        
        objects_fliped[i].pos[0] =  - (objects_fliped[i].pos[0] + 0.06 - 0.27) + (0.27 - 0.06)
        objects_fliped[i].alpha = objects_fliped[i].orientation - m.atan((objects_fliped[i].pos[0] - 0.06) / objects_fliped[i].pos[2])
        if objects_fliped[i].alpha < -m.pi:
            objects_fliped[i].alpha = objects_fliped[i].alpha + 2*m.pi
        if objects_fliped[i].alpha >= m.pi:
            objects_fliped[i].alpha = objects_fliped[i].alpha - 2*m.pi


        exchange_order = [3,2,1,0,7,6,5,4]
        for j in range(8):
            objects_fliped[i].eight_points_in_3d[j] = copy.deepcopy(objects[i].eight_points_in_3d[exchange_order[j]])
            objects_fliped[i].eight_points_in_3d[j][0] = - (objects_fliped[i].eight_points_in_3d[j][0] + 0.06 - 0.27) + (0.27 - 0.06)

            objects_fliped[i].eight_points_in_left[j] = copy.deepcopy(objects[i].eight_points_in_right[exchange_order[j]])
            objects_fliped[i].eight_points_in_left[j][0] = im_shape[1] - 1 - objects_fliped[i].eight_points_in_left[j][0]

            objects_fliped[i].eight_points_in_right[j] = copy.deepcopy(objects[i].eight_points_in_left[exchange_order[j]])
            objects_fliped[i].eight_points_in_right[j][0] = im_shape[1] - 1 - objects_fliped[i].eight_points_in_right[j][0]
        

    return objects, objects_fliped

def load_annotation(index):
    label_file = os.path.join(data_path, 'label_2', index + '.txt')
    calib_file = os.path.join(data_path, 'calib', index + '.txt')

    calib_it = read_obj_calibration(calib_file)
    im_left = load_img_from_index(index,pose = 'left')
    im_right = load_img_from_index(index, pose = 'right')

    img_h, img_w, img_c = im_left.shape
    objects_origin = read_obj_data(label_file, calib_it, im_left.shape) 

    #im_left_flip = cv2.flip(im_right,1)
    #im_right_flip = cv2.flip(im_left, 1)

    #cv2.imwrite(os.path.join(image_2_fliped_path, index + '.png'), im_left)
    #cv2.imwrite(os.path.join(image_2_fliped_path, index + '_flip.png'), im_left_flip)

    #cv2.imwrite(os.path.join(image_3_fliped_path, index + '.png'), im_right)
    #cv2.imwrite(os.path.join(image_3_fliped_path, index + '_flip.png'), im_right_flip)

    objects_origin, objects_fliped = flip_anno(objects_origin, calib_it, im_left.shape)
    write_anno(objects_origin, index, im_left.shape)
    write_anno(objects_fliped, index + '_flip', im_left.shape)


def get_index():
    file_path = os.path.join(data_path,'label_2')
    file_list = os.listdir(file_path)
    index_list = [file.split('.')[0] for file in file_list]
    return index_list

if __name__ == "__main__":
    index_list = get_index()
    for index in index_list:
        print index
        load_annotation(index)
    
    #load_annotation('000986')




