import utils.util as util
import cv2

#检查单张图片，标注框是否准确
if __name__ == "__main__":
    lable_path = r"/home/chenkejing/database/carpetDatabase/Research.v2i.coco/train/labels/20220516_123429_011-jpg_jpg.rf.67387bd3f939c58ef7d2b258e8066419.txt"
    image_path = lable_path.replace("/train/labels/","/train/imgs/").replace(".txt",".jpg")

    # image_path = r"/home/chenkejing/database/carpetDatabase/Annotation.v4i.coco/train/imgs/frame_000210_jpg.rf.fa4214d3747987b7909f827bba6e6f63.jpg"
    # lable_path = image_path.replace("/train/imgs/", "/train/labels/")
    # lable_path = lable_path[:-4] + ".txt"
    ## lable_path = r"/home/chenkejing/database/carpetDatabase/carpet.v1i.coco/train/labels/Screenshot_201_png.rf.2b23b3680a3b419941d96d57450494cb.txt"
    img = util.draw_single_image_yolo_boxes(image_path, lable_path)
    cv2.imwrite("output_with_boxes.jpg", img)
    print("Saved to output_with_boxes.jpg")