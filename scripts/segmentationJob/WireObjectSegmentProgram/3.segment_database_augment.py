"""
@File    : database_augment_seg_safe.py
@Author  : zhangzhuo (modified for YOLOv8-seg, raster-safe cutout)
@Time    : 2026
@Description :
    YOLOv8 Segmentation 数据增强程序
    - 完全兼容 YOLOv8 官方 segmentation label
    - polygon 级别增强（非 bbox）
    - Cutout 安全版本 (raster mask)
"""
from tqdm import tqdm
import cv2
import numpy as np
import random
import math
import os
from glob import glob
from pathlib import Path
from shapely.geometry import Polygon
from shapely.geometry import Point
import utils.util as util

class YOLOSegAugmentor:
    # def __init__(
    #     self,
    #     img_dir,
    #     label_dir,
    #     output_dir,
    #     batch_name,
    #     augment_sample_number=4000,
    #     img_size=(1280, 1280),
    #     long_edge_size=1280,
    #     flip_prob=0.5,        # 水平翻转概率
    #     mosaic_prob=0.2,      # Mosaic 概率（当前未使用）
    #     cutout_prob=0.6,      # Cutout 概率（当前已注释掉）
    #     hsv_prob=0.1,         # HSV 颜色增强概率
    #     # hsv_gain=(0.015, 0.3, 0.25),  # HSV 增强幅度 (H, S, V) Saturation（饱和度）:
    #     #                               # Hue（色相）: H = 0.015 效果：Hue 颜色随机偏移 ±2.7 度 → 颜色轻微变化，不会影响整体色调太多
    #     #                               # S = 0.7 效果：饱和度随机缩放 0.3~1.7 倍；
    #     #                               # V = 0.4 效果：亮度整体缩放 0.6~1.4 倍 → 图像变暗或变亮
    #     hsv_gain=(0.015, 0.3, 0.25),  # 原来是 0.25 → 0.15，整体亮度偏暗
    #     degrees=180.0,         # 仿射旋转角度范围
    #     translate=0.1,        # 平移范围占图像比例
    #     scale=0.1,            # 缩放范围
    #     shear=2.0,            # 剪切角度范围
    # ):
    def __init__(
            self,
            img_dir,
            label_dir,
            output_dir,
            batch_name,
            augment_ratio=2.0,  # ⭐ 改这里
            img_size=(1280, 1280),
            long_edge_size=1280,
            flip_prob=0.5,
            mosaic_prob=0.2,
            cutout_prob=0.6,
            hsv_prob=0.1,
            # hsv_gain=(0.015, 0.3, 0.25),
            hsv_gain=(0.015, 0.4, 0.6),  # 推荐的色彩明暗变化
            degrees=90.0,
            translate=0.1,
            scale=0.1,
            shear=2.0,
    ):
        # 初始化路径和参数
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.output_dir = output_dir
        self.batch_name = batch_name

        # # 获取所有图像文件
        # self.img_files = sorted(
        #     glob(os.path.join(img_dir, "*.jpg")) +
        #     glob(os.path.join(img_dir, "*.jpeg")) +
        #     glob(os.path.join(img_dir, "*.png"))
        # )

        # 获取所有图像文件
        self.img_files = sorted(
            glob(os.path.join(label_dir, "*.txt"))
        )

        # 原始数据数量
        self.num_original = len(self.img_files)

        # ⭐ 倍数参数
        self.augment_ratio = augment_ratio

        # ⭐ 自动计算增强数量
        self.augment_sample_number = int(self.num_original * self.augment_ratio)

        print(f"\n========== 数据增强配置 ==========")
        print(f"原始数据量: {self.num_original}")
        print(f"增强倍数: {self.augment_ratio}")
        print(f"生成样本数: {self.augment_sample_number}")
        print(f"=================================\n")

        self.img_h, self.img_w = img_size
        self.long_edge_size = long_edge_size

        self.flip_prob = flip_prob
        self.mosaic_prob = mosaic_prob
        self.cutout_prob = cutout_prob

        self.hsv_prob = hsv_prob
        self.h_gain, self.s_gain, self.v_gain = hsv_gain

        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

        # 获取所有图像文件
        # self.img_files = sorted(glob(os.path.join(img_dir, "*.jpg")))
        self.img_files = sorted(glob(os.path.join(img_dir, "*.jpg")) +
                                glob(os.path.join(img_dir, "*.jpeg")) +
                                glob(os.path.join(img_dir, "*.png"))
                                )
        # 创建输出文件夹
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

        self.save_index = 0  # 用于生成增强图片序号

    # ------------------------------------------------
    # 读取图像及 YOLOv8 polygon 标签
    # ------------------------------------------------
    def load_image_and_labels(self, index):
        """
        输入索引，读取图像和对应的 polygon 标签
        """
        img_path = self.img_files[index]
        label_path = os.path.join(
            self.label_dir, Path(img_path).stem + ".txt"
        )

        # 读取图像并按最长边缩放
        img = cv2.imread(img_path)
        img = util.resize_long_edge_image(img, self.long_edge_size)
        h, w = img.shape[:2]

        objects = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) < 7:  # 每个 polygon 至少 3 个点 (6 个值)
                        continue
                    cls = int(parts[0])
                    poly = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                    # YOLO 标签归一化 → 图像坐标
                    poly[:, 0] *= w
                    poly[:, 1] *= h
                    if len(poly) >= 3:
                        objects.append({"cls": cls, "poly": poly})

        return img, objects

    # ------------------------------------------------
    # 数据增强函数
    # ------------------------------------------------
    def random_brightness_contrast(self, img, brightness=0.1, contrast=0.15): # brightness 亮度、contrast 对比度
        # 随机亮度/对比度调整
        if random.random() > 0.8:
            return img
        img = img.astype(np.float32)
        alpha = 1.0 + random.uniform(-contrast, contrast)
        beta = random.uniform(-brightness, brightness) * 255
        img = img * alpha + beta
        return np.clip(img, 0, 255).astype(np.uint8)

    def add_gaussian_noise(self, img, mean=0, std=20, prob=0.5):
        # 随机高斯噪声
        if random.random() > prob:
            return img
        noise = np.random.normal(mean, std, img.shape).astype(np.float32)
        img = img.astype(np.float32) + noise
        return np.clip(img, 0, 255).astype(np.uint8)

    def random_hsv(self, img):
        # 随机 HSV 调整
        if random.random() > self.hsv_prob:
            return img
        r = np.random.uniform(-1, 1, 3) * np.array([self.h_gain, self.s_gain, self.v_gain])
        h, s, v = r
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        img[..., 0] = (img[..., 0] + h * 180) % 180
        img[..., 1] *= (1 + s)
        img[..., 2] *= (1 + v)
        img[..., 1:] = np.clip(img[..., 1:], 0, 255)
        return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def random_flip(self, img, objects):
        # 随机水平翻转
        if random.random() < self.flip_prob:
            img = np.fliplr(img).copy()
            w = img.shape[1]
            for obj in objects:
                obj["poly"][:, 0] = w - obj["poly"][:, 0]
        return img, objects

    # ------------------------------------------------
    # Cutout + polygon mask 安全版本
    # ------------------------------------------------
    def cutout_with_mask_raster_safe(
            self, img, objects,
            max_h_ratio=0.4, max_w_ratio=0.4,
            num_holes=(1, 3),
            max_try=50
    ):
        h, w = img.shape[:2]
        num_holes = random.randint(*num_holes)

        # 1️⃣ 创建所有 polygon 的 union mask，用于判断 cutout 是否完全被包围
        union_mask = np.zeros((h, w), dtype=np.uint8)
        for obj in objects:
            pts = obj["poly"].astype(np.int32)
            cv2.fillPoly(union_mask, [pts], 255)

        cutouts = []

        # 2️⃣ 随机生成 cutout 区域
        for _ in range(num_holes):
            for _try in range(max_try):
                ch = random.randint(int(0.05 * h), int(max_h_ratio * h))
                cw = random.randint(int(0.05 * w), int(max_w_ratio * w))
                x = random.randint(0, w - cw)
                y = random.randint(0, h - ch)

                roi = union_mask[y:y + ch, x:x + cw]

                # 如果 cutout 完全在 polygon 内 → 放弃，保证不会被完全包围
                if np.all(roi == 255):
                    continue

                # 相交或相离均可接受
                cutouts.append((x, y, cw, ch))
                # 图像 cutout
                img[y:y + ch, x:x + cw] = np.random.randint(0, 255, (ch, cw, 3), dtype=np.uint8)
                break

        new_objects = []

        # 3️⃣ 对每个 object，生成 mask 并在 mask 上挖洞
        for obj in objects:
            obj_mask = np.zeros((h, w), dtype=np.uint8)
            pts = obj["poly"].astype(np.int32)
            cv2.fillPoly(obj_mask, [pts], 255)

            # 挖洞，确保 cutout 区域在 polygon 中不被标注
            for x, y, cw, ch in cutouts:
                obj_mask[y:y + ch, x:x + cw] = 0

            # 4️⃣ mask → polygon，提取外轮廓（忽略 hole）
            contours, hierarchy = cv2.findContours(
                obj_mask,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if hierarchy is None:
                continue

            for i, cnt in enumerate(contours):
                # 只保留外轮廓（非 hole）
                if hierarchy[0][i][3] != -1:
                    continue
                if len(cnt) >= 3:
                    new_objects.append({
                        "cls": obj["cls"],
                        "poly": cnt.reshape(-1, 2)
                    })

        return img, new_objects

    def random_rotate(self, img, objects):
        """
        随机旋转（polygon 同步变换）
        - 保持目标尽量不裁剪
        - 自动过滤越界 polygon
        """
        if random.random() > 0.35:  # ⭐ 控制旋转概率（建议不要太高）
            return img, objects

        h, w = img.shape[:2]

        # 随机角度（建议小角度更真实）
        angle = random.uniform(-self.degrees, self.degrees)

        # 旋转中心
        cx, cy = w / 2, h / 2

        # 仿射矩阵
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        # 计算新图尺寸（防止裁剪）
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # 平移补偿
        M[0, 2] += (new_w / 2) - cx
        M[1, 2] += (new_h / 2) - cy

        # 图像旋转
        rotated_img = cv2.warpAffine(
            img,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderValue=(114, 114, 114)
        )

        new_objects = []

        # polygon 同步旋转
        for obj in objects:
            poly = obj["poly"]

            # 转齐次坐标
            ones = np.ones((poly.shape[0], 1))
            pts = np.hstack([poly, ones])

            # 仿射变换
            rotated_pts = pts @ M.T

            # 过滤掉完全出界的 polygon
            if (
                    rotated_pts[:, 0].max() < 0 or
                    rotated_pts[:, 1].max() < 0 or
                    rotated_pts[:, 0].min() > new_w or
                    rotated_pts[:, 1].min() > new_h
            ):
                continue

            new_objects.append({
                "cls": obj["cls"],
                "poly": rotated_pts
            })

        return rotated_img, new_objects

    # ------------------------------------------------
    # 保存增强图像及 YOLOv8 segmentation 标签
    # ------------------------------------------------
    def save_sample(self, img, objects):
        h, w = img.shape[:2]
        img_name = f"augment_{self.batch_name}_{self.save_index:06d}.jpg"
        label_name = img_name.replace(".jpg", ".txt")
        # 保存增强后的图像
        cv2.imwrite(os.path.join(self.output_dir, "images", img_name), img)

        # 保存 polygon 标签（带洞）
        with open(os.path.join(self.output_dir, "labels", label_name), "w") as f:
            for obj in objects:
                poly = obj["poly"].astype(np.float32)
                # 转为归一化坐标
                poly[:, 0] /= w
                poly[:, 1] /= h
                poly = np.clip(poly, 0, 1).reshape(-1)
                if len(poly) < 6:
                    continue
                line = " ".join([str(obj["cls"])] + [f"{v:.6f}" for v in poly])
                f.write(line + "\n")

        self.save_index += 1

    # ------------------------------------------------
    # 数据增强 pipeline
    # ------------------------------------------------
    def apply_pipeline(self):
        idx = random.randint(0, len(self.img_files) - 1)
        img, objects = self.load_image_and_labels(idx)

        # ⭐ 先做几何变换（最重要）
        img, objects = self.random_rotate(img, objects)

        # 颜色类增强
        img = self.random_hsv(img)
        img = self.random_brightness_contrast(img)
        img = self.add_gaussian_noise(img, mean=0, std=10, prob=0.5)

        # 空间增强
        img, objects = self.random_flip(img, objects)

        # ⭐ 最后做 cutout（避免影响旋转）
        img, objects = self.cutout_with_mask_raster_safe(img, objects)

        self.save_sample(img, objects)
        return img, objects

# ------------------------------------------------
# 主函数
# ------------------------------------------------
if __name__ == "__main__":
    augmentor = YOLOSegAugmentor(
        img_dir="/home/chenkejing/database/WireDatabase/TotalRealWireDatabase/camera_images_0403_batch1/images",
        label_dir="/home/chenkejing/database/WireDatabase/TotalRealWireDatabase/camera_images_0403_batch1/yolov8_labels/seg",
        output_dir="/home/chenkejing/database/WireDatabase/TotalRealWireDatabase/segment_database_augmentor_0403_batch_1",
        batch_name="segment_real_wire_seg_0403_batch1",
        augment_ratio=3.0
    )

    total = augmentor.augment_sample_number
    with tqdm(total=total, desc="YOLOv8-Seg Augmenting", unit="img", ncols=100, dynamic_ncols=True) as pbar:
        for i in range(total):
            augmentor.apply_pipeline()
            pbar.set_postfix({"saved": augmentor.save_index})
            pbar.update(1)


'''
建议数据增强数量为2~3倍
'''