import os
import json
import cv2
import numpy as np

def extract_ground_mask_bottom_aligned_inpaint(image_path, mask_path, output_path, out_size=640):
    # 读取原图
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR
    h, w = img.shape[:2]

    # 读取 mask polygon
    with open(mask_path, 'r') as f:
        mask_data = json.load(f)
    mask_poly = mask_data[0]['polygons'][0]  # Nx2
    mask_poly = np.array(mask_poly).reshape(-1, 2).astype(np.int32)

    # 创建 mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [mask_poly], 255)

    # 只保留地面区域
    ground = cv2.bitwise_and(img, img, mask=mask)

    # 找到 mask 范围
    ys, xs = np.where(mask > 0)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    crop = ground[y_min:y_max+1, x_min:x_max+1]

    ch, cw = crop.shape[:2]
    canvas_size = max(ch, cw)
    square = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    # 底部对齐放置 crop
    y_start = canvas_size - ch
    x_start = (canvas_size - cw) // 2
    square[y_start:y_start+ch, x_start:x_start+cw] = crop

    # 上方空白区域平铺填充
    for y in range(y_start):
        tile_y = (y % ch)
        square[y, x_start:x_start+cw] = crop[tile_y, :]

    # 左右空白区域平铺填充
    for x in range(x_start):
        tile_x = (x % cw)
        square[:, x] = square[:, x_start + tile_x]
    for x in range(x_start + cw, canvas_size):
        tile_x = (x - (x_start + cw)) % cw
        square[:, x] = square[:, x_start + tile_x]

    # -------------------------
    # Inpainting 无像素区域
    # -------------------------
    # 无像素区域 mask
    gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
    inpaint_mask = (gray == 0).astype(np.uint8) * 255

    if np.any(inpaint_mask):
        # 可选：先模糊边界减少亮度差
        square_blur = cv2.GaussianBlur(square, (5, 5), 0)
        square[inpaint_mask == 255] = square_blur[inpaint_mask == 255]

        # Inpainting
        square = cv2.inpaint(square, inpaint_mask, 3, cv2.INPAINT_NS)

    # resize 到 out_size x out_size
    final_img = cv2.resize(square, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

    # 保存
    cv2.imwrite(output_path, final_img)
    print(f"Saved {output_path}")


# -----------------------------
# 批量处理
# -----------------------------
if __name__ == "__main__":
    images_dir = "/home/chenkejing/database/Floor/floor.v1i.coco/train"
    masks_dir = images_dir.split("train")[0] + "mask_contours"
    output_dir = images_dir.split("train")[0] + "ground_640_inpaint"
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(images_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(images_dir, file)
            mask_path = os.path.join(masks_dir, os.path.splitext(file)[0] + ".json")
            out_path = os.path.join(output_dir, os.path.splitext(file)[0] + "_ground.png")
            if os.path.exists(mask_path):
                extract_ground_mask_bottom_aligned_inpaint(img_path, mask_path, out_path)
