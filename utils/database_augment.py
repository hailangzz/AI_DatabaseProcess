"""
@File    : database_augment.py
@Author  : zhangzhuo
@Time    : 2025/11/13
@Description : 文件功能简述，例如：
              主要的目标检测数据增项程序，mask数据，直接选取最小外接矩形，生成目标检测样本。其中包含各种数据增强操作
@Version : 1.0
"""

import cv2
import numpy as np
import random
import math
import os
import utils.util as util
from glob import glob
from typing import List, Tuple, Optional

class YOLOAugmentor:
    def __init__(self,
                 img_dir,
                 label_dir,
                 img_size: Tuple[int,int] = (1280, 1280),
                 img_longe_eage_size = 1280,
                 hsv_prob: float = 1.0,
                 hsv_gain: Tuple[float,float,float] = (0.015, 0.7, 0.4),
                 exposure_prob=0.6, exposure_range=(0.5, 1.5),
                 noise_prob=0.6, noise_type="both",
                 flip_prob: float = 0.5,
                 mosaic_prob: float = 0.5,
                 mixup_prob: float = 0.0,
                 cutout_prob: float = 0.8,
                 degrees: float = 6.0,
                 translate: float = 0.1,
                 scale: float = 0.1,
                 shear: float = 2.0,
                 ):
        self.img_h, self.img_w = img_size
        self.img_longe_eage_size = img_longe_eage_size
        self.hsv_prob = hsv_prob
        self.exposure_prob = exposure_prob
        self.exposure_range = exposure_range
        self.noise_prob = noise_prob
        self.noise_type = noise_type
        self.h_gain, self.s_gain, self.v_gain = hsv_gain
        self.flip_prob = flip_prob
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.cutout_prob = cutout_prob
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

        self.load_image_paths(img_dir, label_dir)

    def load_image_paths(self, img_dir, label_dir):
        """获取所有图片路径和标签路径"""
        self.img_files = sorted(glob(os.path.join(img_dir, "*.jpg")))
        self.label_files = [
            os.path.join(label_dir, os.path.basename(x).replace(".jpg", ".txt"))
            for x in self.img_files
        ]

    def load_image(self, index):
        """通过索引加载图片和标签"""
        img_path = self.img_files[index]
        label_path = self.label_files[index]

        img = cv2.imread(img_path)
        # 此处就应该做一个图像resize操作，可以大量减少之后的计算算量：
        img = util.resize_long_edge_image(img, target_long=self.img_longe_eage_size)
        assert img is not None, f"Image not found: {img_path}"

        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    cls, x, y, w, h = map(float, line.strip().split())
                    labels.append([cls, x, y, w, h])
        labels = np.array(labels, dtype=np.float32)
        return img, labels


    # ----------------------------- Utilities -----------------------------
    def to_numpy(self, img):
        if isinstance(img, np.ndarray):
            return img
        try:
            import torch
            if isinstance(img, torch.Tensor):
                img = img.permute(1,2,0).cpu().numpy()
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                return img
        except Exception:
            pass
        raise TypeError('Unsupported image type; provide numpy array or torch tensor')

    @staticmethod
    def xyxy_to_yolo(bboxes: np.ndarray, img_shape: Tuple[int,int]):
        """Convert xyxy absolute to YOLO normalized format [class, x_c, y_c, w, h]"""
        h, w = img_shape[:2]
        if len(bboxes) == 0:
            return np.zeros((0,5))
        x1, y1, x2, y2, cls = bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3], bboxes[:,4]
        x_c = (x1 + x2) / 2 / w
        y_c = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        return np.stack([cls, x_c, y_c, bw, bh], axis=1)

    @staticmethod
    def yolo_to_xyxy(bboxes: np.ndarray, img_shape: Tuple[int,int]):
        """Convert YOLO normalized format to xyxy absolute"""
        h, w = img_shape[:2]
        if len(bboxes) == 0:
            return np.zeros((0,5))
        cls, x_c, y_c, bw, bh = bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3], bboxes[:,4]
        x1 = (x_c - bw/2) * w
        y1 = (y_c - bh/2) * h
        x2 = (x_c + bw/2) * w
        y2 = (y_c + bh/2) * h
        return np.stack([x1, y1, x2, y2, cls], axis=1)

    # ----------------------------- Augmentations -----------------------------
    def random_hsv(self, img: np.ndarray):
        # 调节图像亮度对比度
        if random.random() > self.hsv_prob:
            return img
        r = np.random.uniform(-1, 1, 3) * np.array([self.h_gain, self.s_gain, self.v_gain])
        hue, sat, val = r
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        img[...,0] = (img[...,0] + hue * 180) % 180
        img[...,1] = img[...,1] * (1 + sat)
        img[...,2] = img[...,2] * (1 + val)
        img[...,1:3] = np.clip(img[...,1:3], 0, 255)
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return img

    def random_exposure(self, img: np.ndarray):
        """随机调整图像曝光"""
        if random.random() > self.exposure_prob:
            return img

        alpha = random.uniform(*self.exposure_range)  # 线性增益
        gamma = random.uniform(0.8, 1.2)  # γ矫正

        img = np.clip((img / 255.0) ** gamma * alpha * 255.0, 0, 255).astype(np.uint8)
        return img

        # -----------------------------
        # Random Noise
        # -----------------------------
    def random_noise(self, img: np.ndarray):
        """随机添加噪声（高斯 or 椒盐）"""
        if random.random() > self.noise_prob:
            return img

        h, w, c = img.shape
        noisy = img.copy()

        noise_type = self.noise_type
        if noise_type == "both":
            noise_type = random.choice(["gaussian", "saltpepper"])

        # Gaussian noise
        if noise_type == "gaussian":
            mean = 0
            var = random.uniform(10, 30)  # 控制噪声强度
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (h, w, c)).reshape(h, w, c)
            noisy = img + gauss
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        # Salt & Pepper noise
        elif noise_type == "saltpepper":
            s_vs_p = 0.5
            amount = random.uniform(0.002, 0.01)
            num_salt = np.ceil(amount * img.size * s_vs_p)
            num_pepper = np.ceil(amount * img.size * (1.0 - s_vs_p))

            # Salt (white)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape[:2]]
            noisy[coords[0], coords[1], :] = 255

            # Pepper (black)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape[:2]]
            noisy[coords[0], coords[1], :] = 0

        return noisy

    def random_flip(self, img: np.ndarray, bboxes: Optional[np.ndarray]=None):
        # 图像水平旋转功能
        if random.random() < self.flip_prob:
            img = np.fliplr(img).copy()
            if bboxes is not None and len(bboxes):
                # YOLO format [class, x_c, y_c, w, h]
                bboxes[:,1] = 1.0 - bboxes[:,1]
        return img, bboxes

    def random_affine(self, img: np.ndarray, bboxes: np.ndarray = None):
        height, width = img.shape[:2]

        # 若无标注框，直接返回
        if bboxes is None or len(bboxes) == 0:
            return img, bboxes

        # --- 1️⃣ 将 YOLO 坐标转为像素 xyxy ---
        xyxy = self.yolo_to_xyxy(bboxes, img.shape)

        # --- 2️⃣ 构建仿射变换矩阵 ---
        a = random.uniform(-self.degrees, self.degrees)  # 旋转角度
        s = random.uniform(1 - self.scale, 1 + self.scale)  # 缩放
        R = np.eye(3)
        R[:2] = cv2.getRotationMatrix2D(center=(width / 2, height / 2), angle=a, scale=s)

        # 剪切
        S = np.eye(3)
        sx = math.tan(math.radians(random.uniform(-self.shear, self.shear)))
        sy = math.tan(math.radians(random.uniform(-self.shear, self.shear)))
        S[0, 1] = sx
        S[1, 0] = sy

        # 平移
        T = np.eye(3)
        tx = random.uniform(-self.translate, self.translate) * width
        ty = random.uniform(-self.translate, self.translate) * height
        T[0, 2] = tx
        T[1, 2] = ty

        # 合成变换矩阵（顺序：旋转缩放 → 剪切 → 平移）
        M = T @ S @ R

        # --- 3️⃣ 对图像进行仿射变换 ---
        imw = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

        # --- 4️⃣ 变换每个框的四个角点 ---
        n = xyxy.shape[0]
        xy = np.ones((n * 4, 3))
        xy[:, :2] = xyxy[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(n * 4, 2)
        xy = xy @ M.T  # 仿射变换
        xy = xy[:, :2].reshape(n, 8)

        # --- 5️⃣ 用 cv2.minAreaRect 拟合旋转后最小外接矩形 ---
        new_boxes = []
        for i in range(n):
            pts = xy[i].reshape(4, 2).astype(np.float32)
            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect)  # 得到矩形的4个点
            x1, y1 = box[:, 0].min(), box[:, 1].min()
            x2, y2 = box[:, 0].max(), box[:, 1].max()
            new_boxes.append([x1, y1, x2, y2, xyxy[i, 4]])  # 保留类别标签

        new_boxes = np.array(new_boxes)

        # --- 6️⃣ 限制在图像范围内 ---
        new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]].clip(0, width)
        new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]].clip(0, height)

        # --- 7️⃣ 过滤过小框 ---
        keep = (new_boxes[:, 2] - new_boxes[:, 0] > 4) & (new_boxes[:, 3] - new_boxes[:, 1] > 4)
        new_boxes = new_boxes[keep]

        # --- 8️⃣ 转回 YOLO 格式 ---
        bboxes_new = self.xyxy_to_yolo(new_boxes, img.shape)

        return imw, bboxes_new

    # 缩放图像并填充
    def letterbox(self, img: np.ndarray, bboxes: Optional[np.ndarray] = None,
                              new_size: Tuple[int, int] = None, color=(114, 114, 114)):
        if new_size is None:
            new_size = (self.img_h, self.img_w)
        h, w = img.shape[:2]
        ratio = min(new_size[0] / h, new_size[1] / w)
        new_unpad = (int(round(w * ratio)), int(round(h * ratio)))
        dw = new_size[1] - new_unpad[0]
        dh = new_size[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        # 1️⃣ 缩放图像并填充
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(img, int(dh), int(new_size[0] - new_unpad[1] - dh),
                                 int(dw), int(new_size[1] - new_unpad[0] - dw),
                                 cv2.BORDER_CONSTANT, value=color)

        # 2️⃣ 调整YOLO标注
        if bboxes is not None and len(bboxes):
            bboxes = bboxes.copy()

            # 将YOLO归一化坐标转为像素坐标
            bboxes[:, 1] *= w  # x_center
            bboxes[:, 2] *= h  # y_center
            bboxes[:, 3] *= w  # width
            bboxes[:, 4] *= h  # height

            # 应用缩放和偏移
            bboxes[:, 1] = bboxes[:, 1] * ratio + dw
            bboxes[:, 2] = bboxes[:, 2] * ratio + dh
            bboxes[:, 3] = bboxes[:, 3] * ratio
            bboxes[:, 4] = bboxes[:, 4] * ratio

            # 再归一化到新图尺寸
            bboxes[:, 1] /= new_size[1]
            bboxes[:, 2] /= new_size[0]
            bboxes[:, 3] /= new_size[1]
            bboxes[:, 4] /= new_size[0]

        return img, bboxes

    def cutout(self, img: np.ndarray, bboxes: Optional[np.ndarray]=None):
        if random.random() > self.cutout_prob:
            return img, bboxes
        h, w = img.shape[:2]
        num = random.randint(1, 3)
        for _ in range(num):
            ch = random.randint(int(0.02*h), int(0.4*h))
            cw = random.randint(int(0.02*w), int(0.4*w))
            x = random.randint(0, max(0, w - cw))
            y = random.randint(0, max(0, h - ch))
            img[y:y+ch, x:x+cw] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        return img, bboxes

    # ----------------------------- Mosaic -----------------------------
    def mosaic_augment_fixed(self, index):
        # s = self.img_w
        s = self.img_longe_eage_size
        mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        yc, xc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]

        indices = [index] + random.choices(range(len(self.img_files)), k=3)  # 随机在img_files样本集中，抽取3张样本，来进行mosaic数据增强；
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

    # ----------------------------- MixUp -----------------------------
    def mixup(self, img1, labels1, img2, labels2, alpha=0.5):
        r = np.random.beta(alpha, alpha) if alpha > 0 else 1
        img = (img1.astype(np.float32) * r + img2.astype(np.float32) * (1 - r)).astype(np.uint8)
        if len(labels1) and len(labels2):
            labels = np.vstack((labels1, labels2))
        elif len(labels1):
            labels = labels1
        elif len(labels2):
            labels = labels2
        else:
            labels = np.zeros((0,5))
        return img, labels

    # ----------------------------- Pipeline -----------------------------
    def augment(self, img: np.ndarray, labels: Optional[np.ndarray]=None):
        img = self.to_numpy(img)
        labels = np.array(labels, copy=True) if labels is not None else np.zeros((0,5))
        # HSV
        img = self.random_hsv(img)
        # Flip
        img, labels = self.random_flip(img, labels)
        # Cutout
        img, labels = self.cutout(img, labels)
        # Letterbox (keep labels normalized)
        img, labels = self.letterbox(img, labels, new_size=(self.img_h, self.img_w))
        # Convert input xyxy absolute -> YOLO normalized
        # labels = self.xyxy_to_yolo(labels, img.shape)
        # Affine
        img, labels = self.random_affine(img, labels)

        print(labels)
        return img, labels

    def apply_pipeline(self, index):
        """完整增强流程"""
        if random.random() < self.mosaic_prob:
            image, label = self.mosaic_augment_fixed(index)
        else:
            image, label = self.load_image(index)
        image, label = self.random_affine(image, label)
        image, label = self.random_flip(image, label)
        image = self.random_hsv(image)
        image = self.random_exposure(image)
        image = self.random_noise(image)
        image, label = self.cutout(image, label)

        return image, label

# ----------------------------- Example Usage -----------------------------
if __name__ == '__main__':

    augmentor = YOLOAugmentor(img_dir="/home/chenkejing/database/ElectricWiresDataset/test/imgs",
                              label_dir="/home/chenkejing/database/ElectricWiresDataset/test/labels")

    # 随机选择一张图片，执行完整 pipeline
    image, label = augmentor.apply_pipeline(index=0)
    mosaic_img = util.draw_yolo_boxes(image, label, color=(0, 255, 0))
    cv2.imwrite('mosaic_pro.jpg', mosaic_img)
    # quick test
    # aug = YOLOAugmentor(img_size=(640,640), mosaic_prob=0.0, cutout_prob=0.3)
    # img = cv2.imread('test.jpg')  # replace with your path
    # # sample label: x1,y1,x2,y2,class
    # labels = np.array([[0, 0.478750, 0.654359, 0.462500, 0.691281],[0, 0.355750, 0.581628, 0.290000, 0.836744]])
    # out_img, out_labels = aug.augment(img, labels)
    #
    # random_affine = util.draw_yolo_boxes(out_img, out_labels, color=(0, 255, 0))
    # cv2.imwrite('random_hsv.jpg', random_affine)
    # oringin_image = util.draw_yolo_boxes(img, labels, color=(0, 255, 0))
    # cv2.imwrite('oringin_image.jpg', oringin_image)

    # # mosaic usage (requires 4 images and labels)
    # imgs = [cv2.imread(p) for p in ['c1_0.jpg','c1_1.jpg','c1_2.jpg','c1_3.jpg']]
    # labs = [np.array([[0,0.368750,0.500000,0.737500,1.000000]]),
    #         np.array([[0,0.500000,0.842749,1.000000,0.178381],[0,0.500000,0.708407,1.000000,0.090302],[0,0.500000,0.506005,1.000000,0.154359],[0,0.500000,0.228648,1.000000,0.097865],[0,0.500000,0.140125,1.000000,0.170819]]),
    #         np.array([[0,0.500000,0.577847,1.000000,0.844306]]),
    #         np.array([[0,0.460000,0.596530,0.380000,0.806940]])]
    # mosaic_img, mosaic_labels = aug.mosaic(imgs, labs)
    # cv2.imwrite('mosaic.jpg', mosaic_img)
