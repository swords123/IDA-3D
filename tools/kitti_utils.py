from PIL import Image
import numpy as np
import os

class Calibration(object):
    def __init__(self, calib_file_path):
        calib = self.__read_calib(calib_file_path)
        self.P2 = np.reshape(calib['P2'], [3,4])
        self.P3 = np.reshape(calib['P3'], [3,4])

        self.V2C = np.reshape(calib['Tr_velo_to_cam'], [3,4])
        self.C2V = self.__inverse_rigid_trans(self.V2C)

        self.R0 = np.reshape(calib['R0_rect'],[3,3])

        self.c_u = self.P2[0,2]
        self.c_v = self.P2[1,2]
        self.f_u = self.P2[0,0]
        self.f_v = self.P2[1,1]
        self.b_x = self.P2[0,3]/(-self.f_u) # relative 
        self.b_y = self.P2[1,3]/(-self.f_v)


    def __read_calib(self,calib_file_path):
        data = {}
        with open(calib_file_path, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0: continue
                key, value = line.split(':', 1)
                data[key] = np.array([float(x) for x in value.split()])
        return data

    def __inverse_rigid_trans(self, Tr):
        inv_Tr = np.zeros_like(Tr) # 3x4
        inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
        inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
        return inv_Tr

    def __cart2hom(self, pts_3d):
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
        return pts_3d_hom
 
    def project_rect_to_velo(self, pts_3d_rect):
        pts_3d_ref = np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))
        pts_3d_ref = self.__cart2hom(pts_3d_ref)
        pts_3d_velo = np.dot(pts_3d_ref, np.transpose(self.C2V))
        return pts_3d_velo

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_velo = self.__cart2hom(pts_3d_velo)
        pts_3d_ref = np.dot(pts_3d_velo, np.transpose(self.V2C))
        pts_3d_rect = np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))
        return pts_3d_rect

    def project_rect_to_image(self, pts_3d_rect, image_id = 2):
        pts_3d_rect = self.__cart2hom(pts_3d_rect)
        if image_id == 2:
            pts_2d = np.dot(pts_3d_rect, np.transpose(self.P2))
        elif image_id == 3:
            pts_2d = np.dot(pts_3d_rect, np.transpose(self.P3))
        pts_2d[:,0] /= pts_2d[:,2]
        pts_2d[:,1] /= pts_2d[:,2]
        return pts_2d[:,0:2]

    def project_velo_to_image(self, pts_3d_velo, image_id = 2):
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect, image_id)


    def project_image_to_rect(self, uv_depth):
        n = uv_depth.shape[0]
        x = ((uv_depth[:,0]-self.c_u)*uv_depth[:,2])/self.f_u + self.b_x
        y = ((uv_depth[:,1]-self.c_v)*uv_depth[:,2])/self.f_v + self.b_y
        pts_3d_rect = np.zeros((n,3))
        pts_3d_rect[:,0] = x
        pts_3d_rect[:,1] = y
        pts_3d_rect[:,2] = uv_depth[:,2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)


class Object3D(object):
    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        self.type = data[0] 
        self.truncation = data[1] 
        self.occlusion = int(data[2]) # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3] # object observation angle [-pi..pi]

        self.h = data[8] 
        self.w = data[9] 
        self.l = data[10] 
        self.t = (data[11],data[12],data[13]) 
        self.ry = data[14] 

class kitti(object):
    def __init__(self, data_dir, idx):
        self.idx = idx
        self.left_image_dir = os.path.join(data_dir, 'image_2')
        self.right_image_dir = os.path.join(data_dir, 'image_3')
        self.calib_dir = os.path.join(data_dir, 'calib')
        self.lidar_dir = os.path.join(data_dir, 'velodyne')
        self.label_dir = os.path.join(data_dir, 'label_2')
    
    def get_left_img(self):
        img = Image.open(os.path.join(self.left_image_dir, self.idx + '.jpg'))
        img = np.asarray(img)
        return img

    def get_right_img(self):
        img = Image.open(os.path.join(self.right_image_dir,self.idx + '.jpg'))
        img = np.asarray(img)
        return img
    
    def get_lidar(self):
        lidar_filename = os.path.join(self.lidar_dir,self.idx + '.bin')
        scan = np.fromfile(lidar_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return scan

    def get_calib(self):
        calib_filename = os.path.join(self.calib_dir, self.idx +'.txt')
        return Calibration(calib_filename)

    def get_label_objects(self):
        label_filename = os.path.join(self.label_dir, self.idx +'.txt')
        lines = [line.rstrip() for line in open(label_filename)]
        objects = [Object3D(line) for line in lines]
        return objects

def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0):
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin)
    fov_inds = fov_inds & (pc_velo[:,0]>clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds,:]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo



