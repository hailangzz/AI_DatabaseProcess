import os
import json
import cv2
import numpy as np

def extract_ground_mask_y_scaled_inpaint(image_path, mask_path, output_path, out_size=640):
    # 读取原图
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR
    h, w = img.shape[:2]

    # 读取 mask polygon
    with open(mask_path, 'r') as f:
        mask_data = json.load(f)
    mask_poly = mask_data[0]['polygons'][0]  # Nx2
    mask_poly = np.array(mask_poly).reshape(-1, 2).astype(np.int32)

    # 创建 mask
    mask_full = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_full, [mask_poly], 255)

    # 提取地面区域
    ground = cv2.bitwise_and(img, img, mask=mask_full)

    # 裁剪 mask 包围框
    ys, xs = np.where(mask_full > 0)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    crop = ground[y_min:y_max+1, x_min:x_max+1]

    # 等比例缩放到 out_size 高度，仅沿 Y 轴缩放
    ch, cw = crop.shape[:2]
    scale_y = out_size / ch
    new_h = out_size
    new_w = int(cw * scale_y)
    crop_resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 如果宽度超出 out_size，则裁剪中心区域
    if new_w > out_size:
        x_center = new_w // 2
        x_start_crop = x_center - out_size // 2
        crop_resized = crop_resized[:, x_start_crop:x_start_crop + out_size]
        new_w = out_size

    # 保存裁剪后的图像
    cv2.imwrite(output_path, crop_resized)
    print(f"Saved {output_path}")


# -----------------------------
# 批量处理
# -----------------------------
if __name__ == "__main__":
    images_dir = "/home/chenkejing/database/WireDatabase/test0210.v1i.coco/train/imgs"
    masks_dir = images_dir.split("train/")[0] + "mask_contours"
    output_dir = images_dir.split("train/")[0] + "ground_640_final"
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(images_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(images_dir, file)
            mask_path = os.path.join(masks_dir, os.path.splitext(file)[0] + ".json")
            out_path = os.path.join(output_dir, os.path.splitext(file)[0] + "_ground.jpg")
            if os.path.exists(mask_path):
                extract_ground_mask_y_scaled_inpaint(img_path, mask_path, out_path)
