import os
import cv2
import numpy as np
from glob import glob

# -----------------------------
# 读取 YOLOv8 segmentation mask
# -----------------------------
def load_yolo_seg_mask(label_path, img_shape):
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if not os.path.exists(label_path):
        return mask

    with open(label_path) as f:
        for line in f:
            p = list(map(float, line.strip().split()))
            if len(p) < 7 or (len(p) - 1) % 2 != 0:
                continue

            coords = p[1:]
            pts = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * w)
                y = int(coords[i + 1] * h)
                pts.append([x, y])

            pts = np.array(pts, np.int32)
            cv2.fillPoly(mask, [pts], 255)

    return mask


# -----------------------------
# 检测疑似“被填回的 cutout 洞”
# -----------------------------
def detect_filled_holes(img, seg_mask, var_thresh=1500, area_thresh=300):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    var_map = cv2.GaussianBlur((gray - blur) ** 2, (7, 7), 0)
    noise_region = (var_map > var_thresh).astype(np.uint8) * 255

    overlap = cv2.bitwise_and(seg_mask, noise_region)
    contours, _ = cv2.findContours(overlap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    holes = [c for c in contours if cv2.contourArea(c) > area_thresh]
    return holes, noise_region


# -----------------------------
# 可视化函数
# -----------------------------
def visualize_holes(img, seg_mask, holes, alpha=0.4):
    h, w = img.shape[:2]

    # 原图
    vis = img.copy()

    # 红色 mark 区域
    red_layer = np.zeros_like(img)
    red_layer[:, :, 2] = 255
    red_masked = cv2.bitwise_and(red_layer, red_layer, mask=seg_mask)

    # 蓝色非 mark 区域
    inv_mask = cv2.bitwise_not(seg_mask)
    blue_layer = np.zeros_like(img)
    blue_layer[:, :, 0] = 255
    blue_masked = cv2.bitwise_and(blue_layer, blue_layer, mask=inv_mask)

    # 叠加 mask 颜色
    mask_vis = cv2.addWeighted(red_masked, alpha, blue_masked, alpha, 0)
    vis = cv2.addWeighted(vis, 1.0, mask_vis, 1.0, 0)

    # 绘制填回洞
    for c in holes:
        cv2.drawContours(vis, [c], -1, (0, 165, 255), -1)  # 橙色填充
        cv2.drawContours(vis, [c], -1, (0, 0, 255), 2)    # 红色边线

    return vis


# -----------------------------
# 主验证函数
# -----------------------------
def verify_dataset_display(img_dir, label_dir):
    imgs = sorted(glob(os.path.join(img_dir, "*.jpg")))

    for img_path in imgs:
        name = os.path.basename(img_path)
        label_path = os.path.join(label_dir, name.replace(".jpg", ".txt"))

        img = cv2.imread(img_path)
        if img is None:
            continue

        seg_mask = load_yolo_seg_mask(label_path, img.shape)
        holes, _ = detect_filled_holes(img, seg_mask)

        vis = visualize_holes(img, seg_mask, holes)

        cv2.imshow(name, vis)
        key = cv2.waitKey(0)
        cv2.destroyWindow(name)

        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


# -----------------------------
# 用法示例
# -----------------------------
if __name__ == "__main__":
    verify_dataset_display(
        img_dir="/home/chenkejing/database/carpetDatabase/PublicCarpetDatabase_Myself/segment_database_augmentor/images",
        label_dir="/home/chenkejing/database/carpetDatabase/PublicCarpetDatabase_Myself/segment_database_augmentor/labels"
    )

'''
颜色语义非常清晰：
颜色	含义
🔴 红色	segmentation mark 区域
🔵 蓝色	非 mark 区域
🟠 橙色	疑似被 cutout 填回的区域
🔴 红边	洞的轮廓

一个典型能被它抓出来的错误
图像：cutout 填了随机噪声
标注：仍是整块 polygon

👉 你的脚本会把这个区域标成 橙色 + 红边
'''