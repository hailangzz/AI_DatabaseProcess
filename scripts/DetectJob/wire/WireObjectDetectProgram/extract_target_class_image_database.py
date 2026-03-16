import os
import utils.util as util

# 注：此脚本作用是，将test、val等检测集的样本也增加到训练集中（样本较少时，使用增大样本量）

if __name__ == "__main__":

    label_dir = r"/home/chenkejing/database/Wildlife Monitoring and Poaching Detection.v8-final-version/test/labels"
    image_dir = r"/home/chenkejing/database/Wildlife Monitoring and Poaching Detection.v8-final-version/test/imgs"
    output_dir = r"/home/chenkejing/database/Wildlife Monitoring and Poaching Detection.v8-final-version/train/imgs_cls58"

    util.copy_images_by_yolo_labels(label_dir,image_dir,output_dir)