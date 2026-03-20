import utils.util as util

#将项目使用的，各部分样本集，统一存放至一个训练集目录

if __name__ == '__main__':

    part_database_origin_path = r"/data/database/LiquadDatabase/yolo v5_KTH_V2.v1i.coco/valid/"
    src_img_dir = part_database_origin_path+"/imgs"
    src_label_dir = part_database_origin_path+"/labels"

    dst_img_dir = "/data/database/AITotal_ProjectDatabase/LiquadDetectProgrem/images/val"
    dst_label_dir = "/data/database/AITotal_ProjectDatabase/LiquadDetectProgrem/labels/val"

    util.copy_yolo_dataset(src_img_dir, src_label_dir, dst_img_dir, dst_label_dir)