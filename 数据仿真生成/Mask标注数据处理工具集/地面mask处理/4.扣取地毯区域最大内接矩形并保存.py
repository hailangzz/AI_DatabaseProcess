import os
import json
import cv2
import numpy as np


# ==============================
# 最大内接矩形算法（基于最大直方图矩形）
# ==============================
def max_hist_area(heights):
    stack = []
    max_area = 0
    best = (0, 0, 0)  # (start_x, width, height)

    heights.append(0)  # 哨兵
    for i, h in enumerate(heights):
        start = i
        while stack and stack[-1][1] > h:
            index, height = stack.pop()
            area = height * (i - index)
            if area > max_area:
                max_area = area
                best = (index, i - index, height)
            start = index
        stack.append((start, h))
    heights.pop()
    return max_area, best


def find_max_inscribed_rectangle(mask):
    h, w = mask.shape
    height = [0] * w

    max_area = 0
    best_rect = None  # x, y, w, h

    for y in range(h):
        # 更新直方图高度
        for x in range(w):
            if mask[y, x] > 0:
                height[x] += 1
            else:
                height[x] = 0

        area, (sx, rw, rh) = max_hist_area(height.copy())

        if area > max_area:
            max_area = area
            x1 = sx
            x2 = sx + rw - 1
            y2 = y
            y1 = y - rh + 1
            best_rect = (x1, y1, rw, rh)

    return best_rect  # (x, y, width, height)


# ==============================
# 主函数：提取最大内接矩形
# ==============================
def extract_ground_max_rectangle(image_path, mask_path, output_path):
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    # 读取 mask polygon
    with open(mask_path, 'r') as f:
        mask_data = json.load(f)

    if not mask_data:
        print(f"[WARN] No mask data in {mask_path}, skipping.")
        return

    if not mask_data[0].get('polygons'):
        print(f"[WARN] No polygons in {mask_path}, skipping.")
        return

    mask_poly_list = mask_data[0]['polygons'][0]
    if len(mask_poly_list) < 6:
        print(f"[WARN] Polygon too small in {mask_path}, skipping.")
        return

    mask_poly = np.array(mask_poly_list).reshape(-1, 2).astype(np.int32)

    # 创建二值 mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [mask_poly], 255)

    # 找最大内接矩形
    rect = find_max_inscribed_rectangle(mask)
    if rect is None:
        print(f"[WARN] No MIR found for {image_path}")
        return

    x, y, rw, rh = rect
    crop = img[y:y+rh, x:x+rw]

    # 保存裁剪结果
    cv2.imwrite(output_path, crop)
    print(f"Saved {output_path}  Rect = {rect}")


# ==============================
# 批量处理
# ==============================
if __name__ == "__main__":
    images_dir = "/home/chenkejing/database/carpetDatabase/project-w3ryd/project.v3i.coco/train/imgs"
    masks_dir = images_dir.split("train")[0] + "mask_contours"
    output_dir = images_dir.split("train")[0] + "ground_max_rect"
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(images_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(images_dir, file)
            mask_path = os.path.join(masks_dir, os.path.splitext(file)[0] + ".json")
            out_path = os.path.join(output_dir, os.path.splitext(file)[0] + "_max_rect.jpg")

            if os.path.exists(mask_path):
                extract_ground_max_rectangle(img_path, mask_path, out_path)
