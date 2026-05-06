import utils.util as util

#将项目使用的，各部分样本集，统一存放至一个训练集目录

if __name__ == '__main__':

    # part_database_origin_path = r"/data/AITotal_SegmentDatabase/wireDatabaseSegment/"
    # src_img_dir = part_database_origin_path+"/images/train"
    # src_label_dir = part_database_origin_path+"/labels/train"

    part_database_origin_path = r"/home/chenkejing/database/WireDatabase/TotalRealWireDatabase/public_real_camera_images_0422_wire_batch1/"
    src_img_dir = part_database_origin_path + "/images"
    src_label_dir = part_database_origin_path + "/yolov8_labels/seg"

    dst_img_dir = "/data/database/AITotal_SegmentDatabase/finetune_random_sample_datebase/random_wire_database/images/train"
    dst_label_dir = "/data/database/AITotal_SegmentDatabase/finetune_random_sample_datebase/random_wire_database/labels/train"

    util.copy_yolo_dataset(src_img_dir, src_label_dir, dst_img_dir, dst_label_dir)