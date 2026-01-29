import cv2
import numpy as np
import os

def load_yolo_seg_mask(label_path, img_shape):
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if not os.path.exists(label_path):
        return mask

    with open(label_path, "r") as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) < 7:
                continue

            coords = parts[1:]
            if len(coords) % 2 != 0:
                continue

            pts = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * w)
                y = int(coords[i + 1] * h)
                pts.append([x, y])

            pts = np.array(pts, np.int32)
            cv2.fillPoly(mask, [pts], 255)  # 每条 polygon 都填充

    return mask

def show_red_mark_with_holes(image_path, mark_path, alpha=0.5):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图像")

    mark_mask = load_yolo_seg_mask(mark_path, img.shape)

    # 创建红色图层
    red_layer = np.zeros_like(img)
    red_layer[:, :, 2] = 255  # 红色

    # 半透明叠加
    result = img.copy()
    result[mark_mask == 255] = (
        img[mark_mask == 255] * (1 - alpha) +
        red_layer[mark_mask == 255] * alpha
    ).astype(np.uint8)

    cv2.imshow("Red Mark with Holes", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_red_mark_with_holes(
        image_path="/home/chenkejing/database/carpetDatabase/PublicCarpetDatabase_Myself/segment_database_augmentor/images/augment_seg_batch1_000002.jpg",
        mark_path="/home/chenkejing/database/carpetDatabase/PublicCarpetDatabase_Myself/segment_database_augmentor/labels/augment_seg_batch1_000002.txt"
    )
