import os
import json
import cv2
import numpy as np

# ==============================
# 主函数：直接扣取地毯区域最小外接矩形
# ==============================
def extract_ground_bbox(image_path, mask_path, output_path):
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

    # 找最小外接矩形
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        print(f"[WARN] Mask empty in {mask_path}, skipping.")
        return

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # 裁剪图像
    crop = img[y_min:y_max+1, x_min:x_max+1]

    # 保存裁剪结果
    cv2.imwrite(output_path, crop)
    print(f"Saved {output_path}  BBox = ({x_min},{y_min},{x_max},{y_max})")


# ==============================
# 批量处理
# ==============================
if __name__ == "__main__":
    images_dir = "/home/chenkejing/database/carpetDatabase/project-w3ryd/project.v3i.coco/train/imgs"
    masks_dir = images_dir.split("train")[0] + "mask_contours"
    output_dir = images_dir.split("train")[0] + "ground_bbox_crop"
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(images_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(images_dir, file)
            mask_path = os.path.join(masks_dir, os.path.splitext(file)[0] + ".json")
            out_path = os.path.join(output_dir, os.path.splitext(file)[0] + "_bbox.jpg")

            if os.path.exists(mask_path):
                extract_ground_bbox(img_path, mask_path, out_path)
