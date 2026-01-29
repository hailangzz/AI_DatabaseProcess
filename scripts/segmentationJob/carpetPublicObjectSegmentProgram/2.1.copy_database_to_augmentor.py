import utils.util as util

#将项目使用的，各部分样本集，统一存放至一个训练集目录

if __name__ == '__main__':

    part_database_origin_path = r"/home/chenkejing/database/carpetDatabase/PublicCarpetDatabase_Myself/public_carpet_batch1/"
    src_img_dir = part_database_origin_path+"/images"
    src_label_dir = part_database_origin_path+"/yolov8_labels/seg"

    dst_img_dir = "/home/chenkejing/database/carpetDatabase/PublicCarpetDatabase_Myself/origin_public_carpet_database/images"
    dst_label_dir = "/home/chenkejing/database/carpetDatabase/PublicCarpetDatabase_Myself/origin_public_carpet_database/labels"

    util.copy_yolo_dataset(src_img_dir, src_label_dir, dst_img_dir, dst_label_dir)