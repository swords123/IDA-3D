# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
import torch

class TdCoder(object):
    def __init__(self, num_angle_bin = 10, bbox_xform_clip=math.log(1000. / 16)):
        self.reference_whl = [1.5, 1.6, 3.9]

        self.diemsions_weight = [18.0, 18.0, 60.0]
        self.center_weights = [20.0, 12.0]
        self.num_angle_bin = num_angle_bin
        self.rotation_bin = math.pi * 2 / num_angle_bin

        self.bbox_xform_clip = bbox_xform_clip

    def dimentions_encode(self, dimensions):

        wh, ww, wl = self.diemsions_weight

        targets_dh = wh * torch.log(self.reference_whl[0] / dimensions[:, 0])
        targets_dw = ww * torch.log(self.reference_whl[1] / dimensions[:, 1])
        targets_dl = wl * torch.log(self.reference_whl[2] / dimensions[:, 2])

        targets = torch.stack((targets_dh, targets_dw, targets_dl), dim=1)
        return targets

    def dimentions_decode(self, dimensions):
        h, w, l = self.reference_whl
        wh, ww, wl = self.diemsions_weight
        dh = dimensions[:, 0::3] / wh
        dw = dimensions[:, 1::3] / ww
        dl = dimensions[:, 2::3] / wl

        dh = torch.clamp(dh, max=self.bbox_xform_clip)
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dl = torch.clamp(dl, max=self.bbox_xform_clip)

        pred_h = h / torch.exp(dh)
        pred_w = w / torch.exp(dw)
        pred_l = l / torch.exp(dl)

        pred_hwl = torch.zeros_like(dimensions)
        pred_hwl[:, 0::3] = pred_h
        pred_hwl[:, 1::3] = pred_w
        pred_hwl[:, 2::3] = pred_l

        return pred_hwl 

    def dimentions_encode_v2(self, dimensions):
        wh, ww, wl = self.diemsions_weight
        targets_dh = wh * (dimensions[:, 0] - self.reference_whl[0])
        targets_dw = ww * (dimensions[:, 1] - self.reference_whl[1])
        targets_dl = wl * (dimensions[:, 2] - self.reference_whl[2])

        targets = torch.stack((targets_dh, targets_dw, targets_dl), dim=1)
        return targets

    def dimentions_decode_v2(self, dimensions):
        h, w, l = self.reference_whl
        wh, ww, wl = self.diemsions_weight
        dh = dimensions[:, 0::3] / wh
        dw = dimensions[:, 1::3] / ww
        dl = dimensions[:, 2::3] / wl

        dh = torch.clamp(dh, max=self.bbox_xform_clip)
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dl = torch.clamp(dl, max=self.bbox_xform_clip)

        pred_h = h + dh
        pred_w = w + dw
        pred_l = l + dl

        pred_hwl = torch.zeros_like(dimensions)
        pred_hwl[:, 0::3] = pred_h
        pred_hwl[:, 1::3] = pred_w
        pred_hwl[:, 2::3] = pred_l

        return pred_hwl 

    def rotation_y_encode(self, rotation_y):
        rotation_y = rotation_y + math.pi
        rotation_label = rotation_y // self.rotation_bin
        rotation_regression = rotation_y - rotation_label* self.rotation_bin
        rotation_label = rotation_label.long()
        
        return rotation_label, rotation_regression

    def rotation_y_decode(self,rotataion_regress, rotataion_label):
        if(len(rotataion_regress.size())==1):
            rotataion_regress = rotataion_regress.unsqueeze(1)

        rotataion_label = rotataion_label.unsqueeze(1)
        pred_rotation = rotataion_regress + rotataion_label.float() * self.rotation_bin - math.pi
        
        return pred_rotation

    def rotation_y_encode_v2(self, rotation_y):
        rotation_label = (rotation_y + math.pi) // math.pi
        #rotation_regression = torch.abs(rotation_y)
        rotation_regression = (rotation_y + math.pi) % math.pi
        rotation_label = rotation_label.long()
        
        return rotation_label, rotation_regression

    def rotation_y_decode_v2(self,rotataion_regress, rotataion_label):
        if(len(rotataion_regress.size())==1):
            rotataion_regress = rotataion_regress.unsqueeze(1)
        rotataion_label = rotataion_label.unsqueeze(1)
        #pred_rotation = (rotataion_label.float() - 0.5) * 2 * rotataion_regress.float()
        pred_rotation = rotataion_label.float() * math.pi + rotataion_regress.float() - math.pi
        
        return pred_rotation

    def rotation_y_encode_v3(self, rotation_y):
        rotation_y = rotation_y + math.pi
        #rotation_regression = torch.abs(rotation_y)
        angle_label = rotation_y // self.rotation_bin
        angle_label = angle_label.long()
        angle_regression = torch.FloatTensor(angle_label.size()[0], self.num_angle_bin).zero_().cuda()
        for i in range(self.num_angle_bin):
            angle_regression[:,i] = rotation_y - (i + 0.5) * self.rotation_bin
        
        return angle_label, angle_regression

    def rotation_y_decode_v3(self,angle_regression, rotataion_label):
        rotataion_label = rotataion_label.unsqueeze(1)
        channels_num = int(angle_regression.size()[1]/2)
        angle_cos_reg = torch.cos(angle_regression[:,:channels_num])
        angle_sin_reg = torch.sin(angle_regression[:,channels_num:])

        angle_cos_reg = torch.gather(input=angle_cos_reg,dim=1,index=rotataion_label)
        angle_sin_reg = torch.gather(input=angle_sin_reg,dim=1,index=rotataion_label)
        pred_rotation = torch.atan(angle_sin_reg/angle_cos_reg)

        pred_rotation = pred_rotation + (rotataion_label.float()+0.5)*self.rotation_bin - math.pi
        return pred_rotation.squeeze()

    
    def centers_encode(self, centers, proposals):
        TO_REMOVE = 0
        ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
        ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

        gt_ctr_x = centers[:,0]
        gt_ctr_y = centers[:,1]

        wx, wy = self.center_weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets = torch.stack((targets_dx, targets_dy), dim=1)
        
        return targets
    
    def centers_decode(self, codes, boxes): 
        boxes = boxes.to(codes.dtype)

        TO_REMOVE = 0
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy = self.center_weights
        dx = codes[:, 0::2] / wx
        dy = codes[:, 1::2] / wy

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    
        pred_center = torch.zeros_like(codes)
        
        pred_center[:, 0::2] = pred_ctr_x 
        pred_center[:, 1::2] = pred_ctr_y

        return pred_center

    
if __name__ == "__main__":
    c = TdCoder(10)
    '''
    l, r, = c.rotation_y_encode(1.25)
    print(l, r)
    p_r = c.rotation_y_decode(r, l)
    print(p_r)
    '''


