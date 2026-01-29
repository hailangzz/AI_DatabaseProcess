import os
import json
import cv2
import numpy as np


def draw_mask_contours(images_dir, contours_dir, output_dir):
    """
    images_dir: 原始图像目录
    contours_dir: 轮廓标注 JSON 目录
    output_dir: 绘制完成的输出目录
    """
    image_number = 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历每个轮廓文件
    for contour_file in os.listdir(contours_dir):
        if not contour_file.endswith(".json"):
            continue

        img_name = contour_file.replace(".json", "")
        img_path = os.path.join(images_dir, img_name + ".png")  # 假设图片为 PNG
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, img_name + ".jpg")  # 尝试 JPG
            if not os.path.exists(img_path):
                print(f"Image not found for {img_name}, skipping")
                continue

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image {img_path}")
            continue

        # 读取轮廓 JSON
        with open(os.path.join(contours_dir, contour_file), "r") as f:
            contours_data = json.load(f)

        # 绘制每个 mask 轮廓
        for ann in contours_data:
            for poly in ann["polygons"]:
                # poly 是 [x1,y1,x2,y2,...] → 转为 [[x1,y1], [x2,y2], ...]
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)  # 红色线

        # 保存绘制后的图像
        out_path = os.path.join(output_dir, img_name + "_mask_overlay.png")
        cv2.imwrite(out_path, img)
        print(f"Saved overlay image: {out_path}")

        image_number += 1
        if image_number > 2000:
            break


if __name__ == "__main__":
    images_dir = "/home/chenkejing/database/WireDatabase/test0210.v1i.coco/train/imgs"
    # contours_dir = "/home/chenkejing/database/Floor/Texture Detection.v2i.coco/mask_contours"
    #output_dir = "./overlay_images"
    contours_dir = images_dir.split("train/")[0]+"mask_contours"
    output_dir = images_dir.split("train/")[0]+"overlay_images"
    draw_mask_contours(images_dir, contours_dir, output_dir)


