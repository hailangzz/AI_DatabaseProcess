import utils.util as util
import cv2

if __name__ == "__main__":
    image_path = r"/home/chenkejing/database/Cable.v1i/train/imgs/istockphoto-2154844678-2048x2048_cleanup_jpg.rf.b2882fd67256d4e3e9ef89243154b733.jpg"
    lable_path = r"/home/chenkejing/database/Cable.v1i/train/labels/istockphoto-2154844678-2048x2048_cleanup_jpg.rf.b2882fd67256d4e3e9ef89243154b733.txt"
    img = util.draw_single_image_yolo_boxes(image_path,lable_path)
    cv2.imwrite("output_with_boxes.jpg", img)
    print("Saved to output_with_boxes.jpg")