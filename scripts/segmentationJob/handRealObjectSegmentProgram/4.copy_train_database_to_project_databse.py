import utils.util as util

#将项目使用的，各部分样本集，统一存放至一个训练集目录

if __name__ == '__main__':

    part_database_origin_path = r"/home/chenkejing/database/Negativew_Example_Dataset/hand_model_v7/wireDatabaseSegment/segment_database_augmentor_0521_batch2/"
    src_img_dir = part_database_origin_path+"/images"
    src_label_dir = part_database_origin_path+"/labels"

    # part_database_origin_path = r"/home/chenkejing/database/Negativew_Example_Dataset/hand_model_v7/wireDatabaseSegment/"
    # src_img_dir = part_database_origin_path + "/images"
    # src_label_dir = part_database_origin_path + "/yolov8_labels/seg"

    dst_img_dir = "/data/database/AITotal_ProjectDatabase/handDatabaseProgrem/images/train"
    dst_label_dir = "/data/database/AITotal_ProjectDatabase/handDatabaseProgrem/labels/train"

    util.copy_yolo_dataset(src_img_dir, src_label_dir, dst_img_dir, dst_label_dir)