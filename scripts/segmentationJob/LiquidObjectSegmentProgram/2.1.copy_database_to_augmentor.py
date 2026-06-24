import utils.util as util

#将项目使用的，各部分样本集，统一存放至一个训练集目录

if __name__ == '__main__':

    part_database_origin_path = r"/data/database/LiquadDatabase/TotalLiquidDatabase/"
    # src_img_dir = part_database_origin_path+"/imgs"
    # part_database_origin_path = r"/home/chenkejing/database/WireDatabase/origin_publish_wire_database/"
    src_img_dir = part_database_origin_path + "/images"

    src_label_dir = part_database_origin_path+"/yolov8_labels/seg"
    # src_label_dir = part_database_origin_path + "/labels"

    dst_img_dir = "/home/chenkejing/database/AITotal_SegmentDatabase/liquidDatabaseSegment/images/train"
    dst_label_dir = "/home/chenkejing/database/AITotal_SegmentDatabase/liquidDatabaseSegment/labels/train"

    util.copy_yolo_dataset(src_img_dir, src_label_dir, dst_img_dir, dst_label_dir)