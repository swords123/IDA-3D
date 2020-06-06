import torch 
import numpy as np
import os
import kitti_utils  as utils 
import math

pred_path = '/home/pwl/Work/IDA-3D/IDA-3D/self_exp/exp_1/inference/kitti_test/'
data_path = '/home/pwl/Work/IDA-3D/IDA-3D/datasets/kitti/'
checkpoint_item = '62500'

def get_index_from_txt(data_path):
    txt_path = os.path.join(data_path,'splits','test.txt')
    fo = open(txt_path,'r')
    data = fo.read()
    fo.close()
    index_list = data.split('\n')
    return index_list
    
def get_kitti_result(boxes,centers,dimensions,rotations,depth,scores,calib):
    Fu = calib.f_u
    u  = calib.c_u
    Fv = calib.f_v
    v  = calib.c_v
    
    centers_left, centers_right = centers
    
    result = ''
    for idx in range(len(centers_left)):
        
        h = dimensions[idx,0]##
        w = dimensions[idx,1]###
        l = dimensions[idx,2]
        ry = rotations[idx]
        
        center_l = centers_left[idx,:] 
        center_r = centers_right[idx,:] 
        
        #Z = Fu*0.54/abs(depth[idx]) 
        Z = depth[idx]
        X_l = (center_l[0]-u)/Fu*Z + calib.b_x 
        Y_l = (center_l[1]-v)/Fv*Z + calib.b_y  + h/2 
        
        X_r = (center_r[0]-u)/Fu*Z + calib.b_x + 0.54 
        Y_r = (center_r[1]-v)/Fv*Z + calib.b_y  + h/2 

        '''
        X = (X_l + X_r) / 2
        Y = (Y_l + Y_r) / 2
        '''
        X = X_l
        Y = Y_l
        
        rot = ry + math.atan((X-0.06) / Z)
        
        s = scores[idx]
        #if Z<= 87:
        if Z<= 60:
            result += 'Car' + " -1 -1 %f " % (ry)
            result += "%f %f %f %f " % (boxes[idx,0],boxes[idx,1],boxes[idx,2],boxes[idx,3])
            result += "%f %f %f %f %f %f %f %f" % (h,w,l,X,Y,Z,rot,s)
            result += '\n'
    return result

if __name__ == "__main__":

    data_iterm = [checkpoint_item]
    for item in data_iterm:
        data = torch.load(os.path.join(pred_path, "predictions_" + item + ".pth"))
        output_path = os.path.join(pred_path, "result_" + item, "data")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        filenames = get_index_from_txt(data_path)
        for i in range(len(data)):
            index = filenames[i]
            kitti_data = utils.kitti(data_path, index) 
            calib = kitti_data.get_calib()

            boxes = data[i].extra_fields['left_box'].bbox
            scores = data[i].extra_fields['left_box'].extra_fields['scores']
            #centers = data[i].extra_fields['left_centers']
            centers_left = data[i].extra_fields['left_centers']
            centers_right = data[i].extra_fields['right_centers']
        
            dimensions = data[i].extra_fields['dimensions']
            rotations = data[i].extra_fields['rotations']
            depth = data[i].extra_fields['positions_z_depth'][:,0]
            #disp = data[i].extra_fields['positions_z'][:,1]

            result_l = get_kitti_result(boxes,[centers_left, centers_right],dimensions,rotations,depth,scores,calib)
            output_file = os.path.join(output_path,index+'.txt')
            fo = open(output_file,'w')
            fo.write(result_l)
            fo.close()
    

