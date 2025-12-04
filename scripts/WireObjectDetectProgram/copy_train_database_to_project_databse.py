import utils.util as util

#将项目使用的，各部分样本集，统一存放至一个训练集目录

if __name__ == '__main__':

    src_img_dir = "/home/chenkejing/database/Wildlife Monitoring and Poaching Detection.v8-final-version/train/imgs_cls58"
    src_label_dir = "/home/chenkejing/database/Wildlife Monitoring and Poaching Detection.v8-final-version/train/labels"

    dst_img_dir = "/home/chenkejing/database/AITotal_ProjectDatabase/WireDetectProgrem/train/imgs"
    dst_label_dir = "/home/chenkejing/database/AITotal_ProjectDatabase/WireDetectProgrem/train/labels"

    util.copy_yolo_dataset(src_img_dir,src_label_dir,dst_img_dir,dst_label_dir)