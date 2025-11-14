import utils.util as util



if __name__ == '__main__':

    src_img_dir = "/home/chenkejing/database/Wildlife Monitoring and Poaching Detection.v8-final-version/train/imgs_cls58"
    src_label_dir = "/home/chenkejing/database/Wildlife Monitoring and Poaching Detection.v8-final-version/train/labels"

    dst_img_dir = "/home/chenkejing/database/AITotal_ProjectDatabase/WireDetectProgrem/train/imgs"
    dst_label_dir = "/home/chenkejing/database/AITotal_ProjectDatabase/WireDetectProgrem/train/labels"

    util.copy_yolo_dataset(src_img_dir,src_label_dir,dst_img_dir,dst_label_dir)