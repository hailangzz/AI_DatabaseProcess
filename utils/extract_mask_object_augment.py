"""
@File    : extract_mask_object_augment.py
@Author  : zhangzhuo
@Time    : 2025/11/13
@Description : 文件功能简述，例如：
              扣取mask前景，贴合其他图片后，生成目标检测样本的主要程序
@Version : 1.0
"""


import os
import cv2
import numpy as np
import random
import utils.graphic_processing as gp
import utils.util as util

class ExtractMaskObjectAugment:

    def __init__(self,
                 mask_dir, ):
        self.mask_database_origin_path = mask_dir
        self.extract_image_foreground = None
        self.warped_image = None
        self.crop_warped_image = None
        self.augment_image = None
        self.combin_image_yolo = {"image": None,
                                  "yolo_label": [0]}

        self.mask_image_info_dict = {"image_path": None,
                                     "mask_path": None,
                                     "labels_path": None,
                                     "image_name": None}

    def get_mask_image_info(self, image_name):
        self.mask_image_info_dict["image_name"] = image_name
        self.mask_image_info_dict["image_path"] = os.path.join(self.mask_database_origin_path, "imgs", image_name)
        self.mask_image_info_dict["mask_path"] = os.path.join(self.mask_database_origin_path, "masks", image_name)
        self.mask_image_info_dict["labels_path"] = os.path.join(self.mask_database_origin_path, "labels", image_name.split('.')[0]+".txt")

    def apply_pipeline(self,):
        self.extract_image_foreground = gp.extract_object_by_mask_and_yolobox(self.mask_image_info_dict["image_path"],
                                                                              self.mask_image_info_dict["mask_path"],
                                                                              self.mask_image_info_dict["labels_path"])
        self.warped_image = gp.random_perspective_transform(self.extract_image_foreground)
        self.crop_warped_image = gp.crop_min_bounding_rect(self.warped_image)
        self.augment_image = gp.augment_image(self.crop_warped_image)
        self.combin_image_yolo["image"], self.combin_image_yolo["yolo_label"][0] = gp.paste_A_on_B_with_yolo(self.augment_image,"./c1_0.jpg",0)


        gp.save_images("extracted.jpg", self.extract_image_foreground)
        gp.save_images("crop_warped_image.jpg", self.crop_warped_image)
        gp.save_images("augment_image.jpg", self.augment_image)
        gp.save_images("combin_image_yolo.jpg", self.combin_image_yolo["image"])
        util.draw_yolo_boxes(self.combin_image_yolo['image'],self.combin_image_yolo['yolo_label'])

if __name__ == '__main__':
    # 使用示例

    mask_data_origin_path = r"/home/chenkejing/database/ElectricWiresDataset/test"
    extract_mask_image = ExtractMaskObjectAugment(mask_data_origin_path)
    extract_mask_image.get_mask_image_info("c1_10.jpg")
    extract_mask_image.apply_pipeline()


    # extract_mask_image.extract_object_by_mask_and_yolobox(
    #     os.path.join(mask_data_origin_path, "imgs/c1_10.jpg"),
    #     os.path.join(mask_data_origin_path, "masks/c1_10.jpg"),
    #     os.path.join(mask_data_origin_path, "labels/c1_10.txt"),
    #     output_path="./result.jpg"
    # )
    #
    # result = extract_mask_image.paste_object_on_background("c1_2.jpg",
    #                                                        extract_mask_image.extract_image_foreground,
    #                                                        (10, 10),
    #                                                        (0.6, 1.0),
    #                                                        "./combined.jpg")
    #
    # warped_img = extract_mask_image.random_perspective_transform(max_warp_ratio=0.25)
