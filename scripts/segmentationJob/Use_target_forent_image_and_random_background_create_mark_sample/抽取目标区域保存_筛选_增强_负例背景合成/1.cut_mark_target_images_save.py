import os
import cv2
import numpy as np

crop_target_image_num = 0


def crop_yolov8_seg(image_path, label_path, save_dir, min_side_pixels=50):
    global crop_target_image_num

    os.makedirs(save_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        print("image read failed:", image_path)
        return

    h, w = img.shape[:2]

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):

        parts = line.strip().split()

        if len(parts) < 3:
            continue

        cls_id = int(parts[0])
        coords = list(map(float, parts[1:]))

        # 转为像素坐标
        points = []
        for i in range(0, len(coords), 2):

            x = int(coords[i] * w)
            y = int(coords[i + 1] * h)

            points.append([x, y])

        points = np.array(points, dtype=np.int32)

        # 创建mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        # 抠图
        result = cv2.bitwise_and(img, img, mask=mask)

        # 获取bbox
        # 获取bbox
        x, y, bw, bh = cv2.boundingRect(points)

        # ===== 最短边过滤 =====
        min_side = min(bw, bh)
        if min_side < min_side_pixels:
            continue

        # 裁剪图像
        crop_img = img[y:y + bh, x:x + bw]
        crop_mask = mask[y:y + bh, x:x + bw]

        # 创建RGBA图
        rgba = cv2.cvtColor(crop_img, cv2.COLOR_BGR2BGRA)

        # mask作为alpha
        rgba[:, :, 3] = crop_mask

        # 保存
        img_name = os.path.splitext(os.path.basename(image_path))[0]

        save_path = os.path.join(
            save_dir,
            f"{img_name}_{idx}_cls{cls_id}.png"
        )

        cv2.imwrite(save_path, rgba)

        crop_target_image_num += 1

        print("saved:", save_path)


def process_dataset(img_dir, label_dir, save_dir, min_side_pixels=50):

    global crop_target_image_num

    for img_name in os.listdir(img_dir):

        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(img_dir, img_name)

        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)

        if not os.path.exists(label_path):
            continue

        crop_yolov8_seg(
            img_path,
            label_path,
            save_dir,
            min_side_pixels
        )

    print("crop target images number:", crop_target_image_num)


if __name__ == "__main__":

    img_dir = "/home/chenkejing/database/HandDetect/EmdoorRealHandImages/unshare_images/train/images"
    label_dir = "/home/chenkejing/database/HandDetect/EmdoorRealHandImages/unshare_images/train/yolov8_labels/seg"
    save_dir = "/home/chenkejing/database/HandDetect/EmdoorRealHandImages/unshare_images/train/crop_mark_target_image_results"

    # 最短边像素过滤阈值
    process_dataset(
        img_dir,
        label_dir,
        save_dir,
        min_side_pixels=100
    )