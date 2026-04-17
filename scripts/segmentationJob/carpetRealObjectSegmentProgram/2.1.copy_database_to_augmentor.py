import utils.util as util

#将项目使用的，各部分样本集，统一存放至一个训练集目录

if __name__ == '__main__':

    part_database_origin_path = r"/home/chenkejing/database/hard_labels_sample_database/carpet_hard_sample/date0416/segment_database_augmentor_hand_sample_0416_batch_1/"
    src_img_dir = part_database_origin_path+"/images"


    # src_label_dir = part_database_origin_path+"/yolov8_labels/seg"
    src_label_dir = part_database_origin_path + "/labels"

    dst_img_dir = "/data/database/AITotal_SegmentDatabase/finetune_random_sample_datebase/random_carpet_database/images/train"
    dst_label_dir = "/data/database/AITotal_SegmentDatabase/finetune_random_sample_datebase/random_carpet_database/labels/train"

    util.copy_yolo_dataset(src_img_dir, src_label_dir, dst_img_dir, dst_label_dir)