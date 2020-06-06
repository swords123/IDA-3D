import os

import torch
import torch.utils.data
from PIL import Image
import sys
import numpy as np

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.bounding_box import ObjectList

def read_calib(calib_file_path):
    data = {}
    with open(calib_file_path, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line)==0: continue
            key, value = line.split(':', 1)
            data[key] = np.array([float(x) for x in value.split()])
    return data


class KittiDataset(torch.utils.data.Dataset):
    CLASSES = (
        "__background__ ",
        "car",
    )

    def __init__(self, data_dir, split, use_difficult=False, transforms=None):
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms

        self._annopath = os.path.join(self.root, "label_3d", "%s.xml")
        self._image_left_path = os.path.join(self.root, "image_2", "%s.png")
        self._image_right_path = os.path.join(self.root, "image_3", "%s.png")
        self._calib_path = os.path.join(self.root, "calib", "%s.txt")

        self._imgsetpath = os.path.join(self.root, "splits", "%s.txt")

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = KittiDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_left = Image.open(self._image_left_path % img_id).convert("RGB")
        img_right = Image.open(self._image_right_path % img_id).convert("RGB")

        target = self.get_groundtruth(index)
        target_object = self.get_groundtruth(index)
        target_left = target_object.get_field("left_box")
        target_right = target_object.get_field("right_box")
        target_left = target_left.clip_to_image(remove_empty=True)
        target_right = target_right.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img_left, target_left = self.transforms(img_left, target_left)
            img_right, target_right = self.transforms(img_right, target_right)

        target_object.add_field("left_box", target_left)
        target_object.add_field("right_box", target_right)
        calib = self.preprocess_calib(index)

        return img_left, img_right, target, calib, index

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]

        left_target = BoxList(anno["left_boxes"], (width, height), mode="xyxy")
        left_target.add_field("labels", anno["labels"])
        left_target.add_field("difficult", anno["difficult"])

        right_target = BoxList(anno["right_boxes"], (width, height), mode="xyxy")
        right_target.add_field("labels", anno["labels"])
        right_target.add_field("difficult", anno["difficult"])

        object_target = ObjectList()
        object_target.add_field("left_box", left_target)
        object_target.add_field("right_box", right_target)
        object_target.add_field("labels", anno["labels"])
        object_target.add_field("left_centers", anno["left_centers"])
        object_target.add_field("right_centers", anno["right_centers"])
        object_target.add_field("positions_xy", anno["positions_xy"])
        object_target.add_field("positions_z", anno["positions_z"])
        object_target.add_field("dimensions", anno["dimensions"])
        object_target.add_field("alpha", anno["alpha"])
        object_target.add_field("beta", anno["beta"])
        object_target.add_field("corners", anno["corners"])

        assert object_target.is_equal()
        return object_target

    def preprocess_calib(self, index):
        img_id = self.ids[index]
        calib_path = self._calib_path % img_id
        calib = read_calib(calib_path)
        P2 = np.reshape(calib['P2'], [3,4])
        P3 = np.reshape(calib['P3'], [3,4])
        c_u = P2[0,2]
        c_v = P2[1,2]
        f_u = P2[0,0]
        f_v = P2[1,1]
        b_x_2 = P2[0,3]/(f_u) # relative 
        b_y_2 = P2[1,3]/(f_v)
        b_x_3 = P3[0,3]/(f_u) # relative 
        b_y_3 = P3[1,3]/(f_v)
        b = abs(b_x_3 - b_x_2)
        return {
            "cu": c_u, "cv": c_v,
            "fu": f_u, "fv": f_v,
            "b": b, 
            "bx2":b_x_2, 
        }

    def _preprocess_annotation(self, target):
        left_boxes = []
        right_boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 0

        #3d parameters
        left_centers = []
        right_centers = []
        dimensions = []
        positions_xy = []
        positions_z = []
        rotations = []
        alphas = []
        pconers = []
        #occluded = []
        #truncted = []

        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            left_bb = obj.find("left_bndbox")
            left_box = [
                left_bb.find("xmin").text,
                left_bb.find("ymin").text,
                left_bb.find("xmax").text,
                left_bb.find("ymax").text,
            ]
            left_bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(float, left_box)))
            )
            left_boxes.append(left_bndbox)

            left_center = [
                left_bb.find("center").find("x").text, 
                left_bb.find("center").find("y").text,
            ]

            left_center = list(map(float, left_center))
            left_centers.append(left_center)

            right_bb = obj.find("right_bndbox")
            right_box = [
                right_bb.find("xmin").text,
                right_bb.find("ymin").text,
                right_bb.find("xmax").text,
                right_bb.find("ymax").text,
            ]
            right_bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(float, right_box)))
            )
            right_boxes.append(right_bndbox)

            right_center = [
                right_bb.find("center").find("x").text, 
                right_bb.find("center").find("y").text,
            ]

            right_center = list(map(float, right_center))
            right_centers.append(right_center)

            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

            position_xy = [
                obj.find("position").find("x").text,
                obj.find("position").find("y").text,
            ]
            position_xy = list(map(float, position_xy))
            positions_xy.append(position_xy)

            position_z = [
                obj.find("position").find("z").find("depth").text,
                obj.find("position").find("z").find("disp").text,
            ]
            position_z = list(map(float, position_z))
            positions_z.append(position_z)

            dimension = [
                obj.find("dimensions").find("h").text,
                obj.find("dimensions").find("w").text,
                obj.find("dimensions").find("l").text,
            ]
            dimension = list(map(float, dimension))
            dimensions.append(dimension)

            alp = float(obj.find("alpha").text)
            alphas.append(alp)

            rot = float(obj.find("rotation").text)
            rotations.append(rot)

            pc = []
            corners = obj.find("corners")
            for i in range(8):
                pc_str = corners.find("pc%d"%i).text
                pc_i = [float(pc_s) for pc_s in pc_str.split(',')]
                pc.append(pc_i)
            pconers.append(pc)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "left_boxes": torch.tensor(left_boxes, dtype=torch.float32).view(-1,4),
            "right_boxes": torch.tensor(right_boxes, dtype=torch.float32).view(-1,4),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),

            "left_centers": torch.tensor(left_centers, dtype=torch.float32).view(-1,2),
            "right_centers": torch.tensor(right_centers, dtype=torch.float32).view(-1,2),
            "positions_xy": torch.tensor(positions_xy, dtype=torch.float32).view(-1,2),
            "positions_z": torch.tensor(positions_z, dtype=torch.float32).view(-1,2),
            "dimensions": torch.tensor(dimensions, dtype=torch.float32).view(-1,3),
            "alpha": torch.tensor(alphas, dtype=torch.float32),
            "beta": torch.tensor(rotations, dtype=torch.float32),

            "corners": torch.tensor(pconers, dtype=torch.float32).view(-1,8,7),

            "im_info": im_info,

        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return KittiDataset.CLASSES[class_id]
