import os
import cv2
import numpy as np
from tqdm import tqdm


# ==========================================
# 配置
# ==========================================

IMAGE_DIR = "/home/chenkejing/Desktop/WireSegmentProject1/images"
LABEL_DIR = "/home/chenkejing/Desktop/WireSegmentProject1/images"

OUTPUT_VIDEO = "WireSegmentProject_result_0602_1.mp4"

FPS = 5

MASK_ALPHA = 0.4


# ==========================================
# 读取 YOLOv8 Seg
# ==========================================

def load_yolov8_seg(label_file, img_w, img_h):

    polygons = []

    if not os.path.exists(label_file):
        return polygons

    with open(label_file, "r") as f:

        for line in f:

            items = line.strip().split()

            if len(items) < 7:
                continue

            cls_id = int(float(items[0]))

            coords = list(map(float, items[1:]))

            pts = []

            for i in range(0, len(coords), 2):

                x = int(coords[i] * img_w)
                y = int(coords[i + 1] * img_h)

                pts.append([x, y])

            polygons.append(
                (
                    cls_id,
                    np.array(pts, dtype=np.int32)
                )
            )

    return polygons


# ==========================================
# 绘制 Seg
# ==========================================

def draw_segmentation(image, polygons):

    overlay = image.copy()

    color_table = [
        (0, 255, 0),
        (0, 0, 255),
        (255, 0, 0),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255)
    ]

    for cls_id, pts in polygons:

        color = color_table[cls_id % len(color_table)]

        # 填充区域
        cv2.fillPoly(
            overlay,
            [pts],
            color
        )

        # 绘制轮廓
        cv2.polylines(
            image,
            [pts],
            True,
            color,
            2
        )

        # 画点
        for p in pts:
            cv2.circle(
                image,
                tuple(p),
                2,
                (255, 255, 255),
                -1
            )

    cv2.addWeighted(
        overlay,
        MASK_ALPHA,
        image,
        1 - MASK_ALPHA,
        0,
        image
    )

    return image


# ==========================================
# 主程序
# ==========================================

def main():

    image_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])

    if len(image_files) == 0:
        print("No images found.")
        return

    # 读取第一张确定尺寸
    first_img = cv2.imread(
        os.path.join(
            IMAGE_DIR,
            image_files[0]
        )
    )

    h, w = first_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        FPS,
        (w, h)
    )

    for img_name in tqdm(image_files):

        img_path = os.path.join(
            IMAGE_DIR,
            img_name
        )

        label_path = os.path.join(
            LABEL_DIR,
            os.path.splitext(img_name)[0] + ".txt"
        )

        image = cv2.imread(img_path)

        if image is None:
            continue

        polygons = load_yolov8_seg(
            label_path,
            image.shape[1],
            image.shape[0]
        )

        image = draw_segmentation(
            image,
            polygons
        )

        writer.write(image)

    writer.release()

    print()
    print("Video saved:")
    print(OUTPUT_VIDEO)


if __name__ == "__main__":
    main()