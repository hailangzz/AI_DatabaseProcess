import cv2
import os
import random
from pathlib import Path
import numpy as np


def visualize_yolov8_labels(images_dir, bbox_dir=None, seg_dir=None,
                            num_samples=5, save_dir=None, alpha=0.4):
    """
    随机抽取图片，并在图片上绘制 YOLOv8 bbox / seg 标签进行可视化
    分割区域使用半透明红色遮罩

    参数：
        images_dir: 图片文件夹
        bbox_dir: bbox txt 标签文件夹（可选）
        seg_dir: seg txt 标签文件夹（可选）
        num_samples: 随机抽取图片数量
        save_dir: 可选，保存可视化结果
        alpha: 分割区域透明度 (0~1)
    """
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    images_dir = os.path.join(images_dir, "images")
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not image_files:
        print("[WARN] 没有找到图片")
        return

    samples = random.sample(image_files, min(num_samples, len(image_files)))

    for img_file in samples:
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        base_name = Path(img_file).stem

        overlay = img.copy()  # 用于绘制半透明遮罩

        # ---------- 绘制 bbox ----------
        if bbox_dir:
            bbox_file = os.path.join(bbox_dir, base_name + ".txt")
            if os.path.exists(bbox_file):
                with open(bbox_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        class_id, x_c, y_c, bw, bh = map(float, parts)
                        x1 = int((x_c - bw / 2) * w)
                        y1 = int((y_c - bh / 2) * h)
                        x2 = int((x_c + bw / 2) * w)
                        y2 = int((y_c + bh / 2) * h)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, str(int(class_id)), (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ---------- 绘制 segmentation ----------
        if seg_dir:
            seg_file = os.path.join(seg_dir, base_name + ".txt")
            if os.path.exists(seg_file):
                with open(seg_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 3:
                            continue
                        class_id = int(parts[0])
                        coords = list(map(float, parts[1:]))
                        pts = []
                        for i in range(0, len(coords), 2):
                            x = int(coords[i] * w)
                            y = int(coords[i + 1] * h)
                            pts.append([x, y])
                        pts = np.array(pts, np.int32).reshape((-1, 1, 2))

                        # 在 overlay 上填充半透明红色
                        cv2.fillPoly(overlay, [pts], color=(0, 0, 255))
                        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 0, 180), thickness=2)
                        cv2.putText(overlay, str(class_id), tuple(pts[0][0]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

                # 叠加半透明
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # ---------- 显示或保存 ----------
        if save_dir:
            save_path = os.path.join(save_dir, img_file)
            cv2.imwrite(save_path, img)
        else:
            cv2.imshow("YOLOv8 Labels", img)
            key = cv2.waitKey(0)
            if key == 27:  # ESC
                break
    cv2.destroyAllWindows()


# --------------------------
# 测试调用d
# --------------------------
if __name__ == "__main__":
    visualize_yolov8_labels(
        images_dir="/home/chenkejing/database/carpetDatabase/PublicCarpetDatabase_Myself/public_carpet_batch1/",
        bbox_dir="/home/chenkejing/database/carpetDatabase/PublicCarpetDatabase_Myself/public_carpet_batch1/yolov8_labels/bbox",
        seg_dir="/home/chenkejing/database/carpetDatabase/PublicCarpetDatabase_Myself/public_carpet_batch1/yolov8_labels/seg",
        num_samples=70,
        save_dir=None,  # 设置文件夹路径可保存结果
        alpha=0.4  # 半透明程度
    )

    # visualize_yolov8_labels(
    #     images_dir="/home/chenkejing/database/carpetDatabase/EMdoorRealCarpetDatabase/segment_database_augmentor",
    #     bbox_dir="/home/chenkejing/database/carpetDatabase/EMdoorRealCarpetDatabase/segment_database_augmentor/images",
    #     seg_dir="/home/chenkejing/database/carpetDatabase/EMdoorRealCarpetDatabase/segment_database_augmentor/labels",
    #     num_samples=20,
    #     save_dir=None,  # 设置文件夹路径可保存结果
    #     alpha=0.4  # 半透明程度
    # )