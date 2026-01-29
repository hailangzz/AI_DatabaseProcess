import utils.util as util

#将项目使用的，各部分样本集，统一存放至一个训练集目录

if __name__ == '__main__':

    # part_database_origin_path = r"/home/chenkejing/database/carpetDatabase/EMdoorRealCarpetDatabase/origin_real_carpet_database/"
    # src_img_dir = part_database_origin_path+"/images/train"
    # src_label_dir = part_database_origin_path+"/labels/train"

    part_database_origin_path = r"/home/chenkejing/database/carpetDatabase/EMdoorRealCarpetDatabase/segment_database_augmentor/"
    src_img_dir = part_database_origin_path + "/images"
    src_label_dir = part_database_origin_path + "/labels"

    dst_img_dir = "/home/chenkejing/database/AITotal_SegmentDatabase/carpetDatabaseSegment/images/train"
    dst_label_dir = "/home/chenkejing/database/AITotal_SegmentDatabase/carpetDatabaseSegment/labels/train"

    util.copy_yolo_dataset(src_img_dir, src_label_dir, dst_img_dir, dst_label_dir)