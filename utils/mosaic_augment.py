import os
import cv2
import random
import numpy as np
from glob import glob
import utils.util as util

os.environ["QT_QPA_PLATFORM"] = "offscreen"

class DataAugmentor:
    def __init__(self, img_dir, label_dir, img_size=640):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size

        # 获取所有图片路径
        self.img_files = sorted(glob(os.path.join(img_dir, "*.jpg")))
        self.label_files = [
            os.path.join(label_dir, os.path.basename(x).replace(".jpg", ".txt"))
            for x in self.img_files
        ]

    def load_image(self, index):
        """加载单张图片与对应标签"""
        img_path = self.img_files[index]
        label_path = self.label_files[index]

        img = cv2.imread(img_path)
        assert img is not None, f"Image not found: {img_path}"

        h, w = img.shape[:2]
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    cls, x, y, bw, bh = map(float, line.strip().split())
                    labels.append([cls, x, y, bw, bh])
        labels = np.array(labels, dtype=np.float32)
        print(labels.shape)
        return img, labels

    def mosaic_augment_fixed(self, index):
        s = self.img_size
        mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        yc, xc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]

        indices = [index] + random.choices(range(len(self.img_files)), k=3)
        mosaic_labels = []

        for i, idx in enumerate(indices):
            img, labels = self.load_image(idx)
            h, w = img.shape[:2]
            scale = s / max(h, w)
            if random.random() < 0.5:
                scale *= random.uniform(0.5, 1.5)
            nh, nw = int(h * scale), int(w * scale)
            img = cv2.resize(img, (nw, nh))

            # 放入 mosaic 画布中的位置
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - nw, 0), max(yc - nh, 0), xc, yc
                x1b, y1b, x2b, y2b = nw - (x2a - x1a), nh - (y2a - y1a), nw, nh
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - nh, 0), min(xc + nw, s * 2), yc
                x1b, y1b, x2b, y2b = 0, nh - (y2a - y1a), min(nw, x2a - x1a), nh
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - nw, 0), yc, xc, min(s * 2, yc + nh)
                x1b, y1b, x2b, y2b = nw - (x2a - x1a), 0, nw, min(y2a - y1a, nh)
            else:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + nw, s * 2), min(s * 2, yc + nh)
                x1b, y1b, x2b, y2b = 0, 0, min(nw, x2a - x1a), min(nh, y2a - y1a)

            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

            # --- 修正标注 ---
            if labels is not None and len(labels):
                labels = labels.copy()
                # YOLO -> 绝对坐标
                labels[:, 1] = labels[:, 1] * w * scale
                labels[:, 2] = labels[:, 2] * h * scale
                labels[:, 3] = labels[:, 3] * w * scale
                labels[:, 4] = labels[:, 4] * h * scale

                # 加平移
                labels[:, 1] = labels[:, 1] - x1b + x1a
                labels[:, 2] = labels[:, 2] - y1b + y1a

                mosaic_labels.append(labels)

        # === 合并所有标签 ===
        if len(mosaic_labels):
            mosaic_labels = np.concatenate(mosaic_labels, axis=0)

            # 将中心点、宽高 → xyxy
            xyxy = np.zeros_like(mosaic_labels)
            xyxy[:, 0] = mosaic_labels[:, 0]  # class
            xyxy[:, 1] = mosaic_labels[:, 1] - mosaic_labels[:, 3] / 2
            xyxy[:, 2] = mosaic_labels[:, 2] - mosaic_labels[:, 4] / 2
            xyxy[:, 3] = mosaic_labels[:, 1] + mosaic_labels[:, 3] / 2
            xyxy[:, 4] = mosaic_labels[:, 2] + mosaic_labels[:, 4] / 2

            # === 裁剪到最终区域 ===
            x1 = max(xc - s // 2, 0)
            y1 = max(yc - s // 2, 0)
            x2 = x1 + s
            y2 = y1 + s
            mosaic_img = mosaic_img[y1:y2, x1:x2]

            # 将框裁剪到边界（保留部分可见目标）
            xyxy[:, 1] = np.clip(xyxy[:, 1] - x1, 0, s)
            xyxy[:, 2] = np.clip(xyxy[:, 2] - y1, 0, s)
            xyxy[:, 3] = np.clip(xyxy[:, 3] - x1, 0, s)
            xyxy[:, 4] = np.clip(xyxy[:, 4] - y1, 0, s)

            # 过滤过小或无效框
            w_box = xyxy[:, 3] - xyxy[:, 1]
            h_box = xyxy[:, 4] - xyxy[:, 2]
            keep = (w_box > 2) & (h_box > 2)
            xyxy = xyxy[keep]

            # === 回 YOLO 格式 ===
            if len(xyxy):
                labels_new = np.zeros_like(xyxy)
                labels_new[:, 0] = xyxy[:, 0]
                labels_new[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2 / s
                labels_new[:, 2] = (xyxy[:, 2] + xyxy[:, 4]) / 2 / s
                labels_new[:, 3] = w_box[keep] / s
                labels_new[:, 4] = h_box[keep] / s
            else:
                labels_new = np.zeros((0, 5))
        else:
            labels_new = np.zeros((0, 5))

        return mosaic_img, labels_new


# ==== 示例调用 ====
if __name__ == "__main__":
    # 假设你有以下结构：
    # dataset/
    # ├── images/
    # │    ├── 0001.jpg
    # │    ├── 0002.jpg
    # ├── labels/
    #      ├── 0001.txt
    #      ├── 0002.txt

    augmentor = DataAugmentor(
        img_dir="/home/chenkejing/database/ElectricWiresDataset/test/imgs",
        label_dir="/home/chenkejing/database/ElectricWiresDataset/test/labels",
        img_size=640
    )

    # 随机取一张做 mosaic 增强
    idx = random.randint(0, len(augmentor.img_files) - 1)
    mosaic_img, mosaic_labels = augmentor.mosaic_augment_fixed(idx)

    print("增强后标签 shape:", mosaic_labels.shape)

    # === 可视化 ===
    img_vis = mosaic_img.copy()
    for lbl in mosaic_labels:
        cls, x, y, w, h = lbl
        h_img, w_img = img_vis.shape[:2]
        x1 = int((x - w / 2) * w_img)
        y1 = int((y - h / 2) * h_img)
        x2 = int((x + w / 2) * w_img)
        y2 = int((y + h / 2) * h_img)
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_vis, f"{int(cls)}", (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


    mosaic_img = util.draw_yolo_boxes(mosaic_img, mosaic_labels, color=(0, 255, 0))
    cv2.imwrite('mosaic.jpg', mosaic_img)