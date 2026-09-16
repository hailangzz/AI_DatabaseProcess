# -*- coding: utf-8 -*-

"""
Hard Case Dataset 信息扫描 + 数据集生成程序

功能：

第一部分：
    扫描 LOCAL_SAVE_DIR 下所有解压后的 Hard Case 数据集，
    生成：

        LOCAL_SAVE_DIR/hard_case_info.json


第二部分：
    根据 hard_case_info.json 自动生成两个数据集：

        exist_target_dataset/
        ├── images/
        └── labels/

        unexist_target_dataset/
        ├── images/
        └── labels/


其中：

exist_target_dataset：
    保存有标注的图片及其对应的 label。

unexist_target_dataset：
    从没有标注的图片中随机抽取 1/3，
    保存图片，并创建同名空 txt label。


最终保证：

    exist_target_dataset/images/xxx.jpg
    exist_target_dataset/labels/xxx.txt

    一一对应


以及：

    unexist_target_dataset/images/xxx.jpg
    unexist_target_dataset/labels/xxx.txt

    一一对应。
"""

import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import utils.util as util

# ============================================================
# 支持的图片格式
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# 工具函数
# ============================================================


def to_relative_path(file_path: str, base_dir: str):
    """
    将绝对路径转换为相对于 base_dir 的路径。
    """

    relative_path = os.path.relpath(file_path, base_dir)

    return relative_path.replace(os.sep, "/")


# ============================================================
# 扫描单个数据集
# ============================================================


def scan_dataset(dataset_dir: str, local_save_dir: str):
    """
    扫描一个解压后的数据集目录。

    例如：

        dataset_dir/
        ├── data.yaml
        ├── images/
        │   ├── train/
        │   │   ├── 001.jpg
        │   │   └── 002.jpg
        │   │
        │   └── val/
        │       └── 003.jpg
        │
        └── labels/
            ├── train/
            │   ├── 001.txt
            │   └── 002.txt
            │
            └── val/
                └── 003.txt
    """

    dataset_name = os.path.basename(os.path.normpath(dataset_dir))

    images_dir = os.path.join(dataset_dir, "images")

    labels_dir = os.path.join(dataset_dir, "labels")

    print("\n" + "-" * 80)
    print(f"数据集：{dataset_name}")
    print(f"路径：  {dataset_dir}")
    print("-" * 80)

    # ========================================================
    # 检查 images
    # ========================================================

    if not os.path.isdir(images_dir):
        print("[WARNING] images目录不存在，跳过该数据集。")

        return {
            "images": [],
            "labels": [],
            "labeled_images": [],
            "unlabeled_images": [],
        }

    # ========================================================
    # 检查 labels
    # ========================================================

    if not os.path.isdir(labels_dir):
        print("[WARNING] labels目录不存在。")

    # ========================================================
    # 扫描图片
    # ========================================================

    image_files = []

    for root, _, files in os.walk(images_dir):

        for filename in files:

            extension = os.path.splitext(filename)[1].lower()

            if extension not in IMAGE_EXTENSIONS:
                continue

            full_path = os.path.join(root, filename)

            relative_path = to_relative_path(full_path, local_save_dir)

            image_files.append(relative_path)

    # ========================================================
    # 扫描 Label
    # ========================================================

    label_files = []

    if os.path.isdir(labels_dir):

        for root, _, files in os.walk(labels_dir):

            for filename in files:

                if not filename.lower().endswith(".txt"):
                    continue

                full_path = os.path.join(root, filename)

                relative_path = to_relative_path(full_path, local_save_dir)

                label_files.append(relative_path)

    # ========================================================
    # 建立当前数据集的 Label 映射
    # ========================================================

    label_map = {}

    duplicate_labels = []

    for label_path in label_files:

        filename = os.path.basename(label_path)

        stem = os.path.splitext(filename)[0]

        if stem in label_map:

            duplicate_labels.append(label_path)

        else:

            label_map[stem] = label_path

    # ========================================================
    # 图片分类
    # ========================================================

    labeled_images = []

    unlabeled_images = []

    for image_path in image_files:

        filename = os.path.basename(image_path)

        image_stem = os.path.splitext(filename)[0]

        if image_stem in label_map:

            labeled_images.append(image_path)

        else:

            unlabeled_images.append(image_path)

    # ========================================================
    # 排序
    # ========================================================

    image_files.sort()
    label_files.sort()
    labeled_images.sort()
    unlabeled_images.sort()

    # ========================================================
    # 输出当前数据集统计
    # ========================================================

    print(f"图片数量：          {len(image_files)}")

    print(f"Label数量：         {len(label_files)}")

    print(f"有标注图片：        {len(labeled_images)}")

    print(f"无标注图片：        {len(unlabeled_images)}")

    if duplicate_labels:

        print(f"[WARNING] 发现 " f"{len(duplicate_labels)} " f"个重复 label basename。")

        for path in duplicate_labels[:10]:
            print(f"    {path}")

    return {
        "images": image_files,
        "labels": label_files,
        "labeled_images": labeled_images,
        "unlabeled_images": unlabeled_images,
    }


# ============================================================
# 生成 hard_case_info.json
# ============================================================


def generate_hard_case_info(local_save_dir: str):
    """
    扫描 LOCAL_SAVE_DIR 下所有数据集，
    生成 hard_case_info.json。
    """

    print("\n" + "=" * 80)
    print("开始扫描 Hard Case 数据集")
    print("=" * 80)

    if not os.path.isdir(local_save_dir):
        raise FileNotFoundError(f"LOCAL_SAVE_DIR 不存在：{local_save_dir}")

    # ========================================================
    # 获取所有一级目录
    # ========================================================

    dataset_dirs = []

    for name in sorted(os.listdir(local_save_dir)):

        path = os.path.join(local_save_dir, name)

        if not os.path.isdir(path):
            continue

        # 不扫描程序生成的目标目录
        if name in {
            "exist_target_dataset",
            "unexist_target_dataset",
        }:
            continue

        dataset_dirs.append(path)

    print(f"发现数据集目录：{len(dataset_dirs)} 个")

    # ========================================================
    # 汇总
    # ========================================================

    all_labeled_images = []

    all_unlabeled_images = []

    all_labels = []

    processed_dataset_count = 0

    # ========================================================
    # 扫描数据集
    # ========================================================

    for dataset_dir in dataset_dirs:

        result = scan_dataset(dataset_dir, local_save_dir)

        if not result["images"]:
            continue

        processed_dataset_count += 1

        all_labeled_images.extend(result["labeled_images"])

        all_unlabeled_images.extend(result["unlabeled_images"])

        all_labels.extend(result["labels"])

    # ========================================================
    # 排序
    # ========================================================

    all_labeled_images.sort()

    all_unlabeled_images.sort()

    all_labels.sort()

    # ========================================================
    # 统计
    # ========================================================

    image_count = len(all_labeled_images) + len(all_unlabeled_images)

    label_count = len(all_labels)

    labeled_image_count = len(all_labeled_images)

    unlabeled_image_count = len(all_unlabeled_images)

    # ========================================================
    # JSON
    # ========================================================

    hard_case_info = {
        "image_count": image_count,
        "label_count": label_count,
        "labeled_image_count": labeled_image_count,
        "unlabeled_image_count": unlabeled_image_count,
        "labeled_images": all_labeled_images,
        "unlabeled_images": all_unlabeled_images,
        "labels": all_labels,
    }

    # ========================================================
    # 输出 JSON
    # ========================================================

    with open(HARD_CASE_INFO_JSON, "w", encoding="utf-8") as f:

        json.dump(hard_case_info, f, ensure_ascii=False, indent=4)

    # ========================================================
    # 输出统计
    # ========================================================

    print("\n" + "=" * 80)
    print("Hard Case 信息生成完成")
    print("=" * 80)

    print(f"处理数据集数量：      " f"{processed_dataset_count}")

    print(f"所有图片数量：        " f"{image_count}")

    print(f"所有 Label 数量：      " f"{label_count}")

    print(f"有标注图片数量：      " f"{labeled_image_count}")

    print(f"无标注图片数量：      " f"{unlabeled_image_count}")

    print(f"JSON 文件：           " f"{HARD_CASE_INFO_JSON}")

    print("=" * 80)


# ============================================================
# Hard Case 样本集数据增强
# ============================================================

class YOLOSegAugmentor:
    """
    对 TARGET_DATASET/exist_target_dataset 进行离线数据增强。

    输出：
        TARGET_DATASET/
        └── Augmentor_exist_target_dataset/
            ├── images/
            └── labels/

    命名：
        原图：xxx.jpg
        增强：Augmentor_xxx_001.jpg
              Augmentor_xxx_002.jpg
              ...

        对应 Label：
        Augmentor_xxx_001.txt
        Augmentor_xxx_002.txt
        ...

    说明：
        - 保留原始 exist_target_dataset 不变。
        - 只读取 exist_target_dataset/images + labels。
        - 使用 YOLOv8-seg polygon 标签进行同步几何变换。
        - 每张原始 Hard Case 默认生成 3 张增强图。
        - 如果某张图片增强后目标全部消失，则重新生成，避免产生无目标增强样本。
        - 空 Label 文件属于合法样本：
            * 图片正常进行增强
            * 增强后的 Label 生成空 txt
    """

    def __init__(
            self,
            img_dir,
            label_dir,
            output_dir,
            augment_ratio=3.0,
            long_edge_size=1280,
            flip_prob=0.5,
            hsv_prob=0.1,
            hsv_gain=(0.015, 0.4, 0.6),
            degrees=90.0,
    ):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.output_dir = output_dir

        self.augment_ratio = float(augment_ratio)

        if self.augment_ratio <= 0:
            raise ValueError("augment_ratio 必须 > 0")

        # 这里采用“每张原图生成 N 张增强图”的方式。
        # 例如 3.0 -> 每张原图生成 3 张。
        self.augment_per_image = int(self.augment_ratio)

        if self.augment_per_image < 1:
            self.augment_per_image = 1

        self.long_edge_size = long_edge_size
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.h_gain, self.s_gain, self.v_gain = hsv_gain
        self.degrees = degrees

        self.img_files = sorted(
            [
                os.path.join(img_dir, f)
                for f in os.listdir(img_dir)
                if os.path.splitext(f)[1].lower()
                   in IMAGE_EXTENSIONS
            ]
        )

        if not self.img_files:
            raise FileNotFoundError(
                f"没有找到可增强的图片：{img_dir}"
            )

        os.makedirs(
            os.path.join(output_dir, "images"),
            exist_ok=True
        )

        os.makedirs(
            os.path.join(output_dir, "labels"),
            exist_ok=True
        )

        self.success_count = 0
        self.failed_count = 0

        print("\n" + "=" * 80)
        print("Hard Case 样本增强配置")
        print("=" * 80)
        print(f"输入图片目录：      {self.img_dir}")
        print(f"输入 Label 目录：   {self.label_dir}")
        print(f"输出目录：          {self.output_dir}")
        print(f"原始样本数量：      {len(self.img_files)}")
        print(f"每张图片增强数量：  {self.augment_per_image}")
        print(
            f"预计增强样本数量：  "
            f"{len(self.img_files) * self.augment_per_image}"
        )
        print("=" * 80)

    def resize_to_max_size(self, img, objects, max_size=640):
        """
        等比例缩放图片，使最长边不超过 max_size。

        同时同步缩放 polygon 像素坐标。

        注意：
            YOLO 最终保存的是归一化坐标，
            所以同步缩放后，最终归一化坐标理论上保持不变。
        """

        h, w = img.shape[:2]

        max_edge = max(h, w)

        if max_edge <= max_size:
            return img, objects

        scale = max_size / max_edge

        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized_img = cv2.resize(
            img,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

        # 同步缩放 polygon 像素坐标
        for obj in objects:
            obj["poly"][:, 0] *= scale
            obj["poly"][:, 1] *= scale

        return resized_img, objects

    def load_image_and_labels(self, img_path):
        """读取图片和 YOLOv8-seg polygon 标签。"""

        image_name = os.path.basename(img_path)
        stem = os.path.splitext(image_name)[0]

        label_path = os.path.join(
            self.label_dir,
            stem + ".txt"
        )

        img = cv2.imread(img_path)

        if img is None:
            raise RuntimeError(
                f"无法读取图片：{img_path}"
            )

        img = util.resize_long_edge_image(
            img,
            self.long_edge_size
        )

        h, w = img.shape[:2]

        objects = []

        if not os.path.isfile(label_path):
            raise FileNotFoundError(
                f"找不到对应 Label：{label_path}"
            )

        # ========================================================
        # 新增逻辑：
        # 空 Label 文件属于合法样本。
        #
        # 如果 txt 文件存在，但大小为 0：
        #     objects = []
        #     返回空对象列表
        #
        # run() 会继续执行增强，而不是跳过。
        #
        # 注意：
        # 这里不把“空文件”当成错误。
        # ========================================================
        if os.path.getsize(label_path) == 0:
            return img, objects, True

        with open(
                label_path,
                "r",
                encoding="utf-8"
        ) as f:

            for line in f:
                line = line.strip()

                if not line:
                    continue

                parts = list(
                    map(float, line.split())
                )

                # class + 至少 3 个 polygon 点
                if len(parts) < 7:
                    continue

                cls = int(parts[0])
                coords = parts[1:]

                if len(coords) % 2 != 0:
                    continue

                poly = np.array(
                    coords,
                    dtype=np.float32
                ).reshape(-1, 2)

                poly[:, 0] *= w
                poly[:, 1] *= h

                if len(poly) >= 3:
                    objects.append({
                        "cls": cls,
                        "poly": poly
                    })

        # 非空 Label，但没有解析出有效 polygon。
        #
        # 这里仍然返回 False，
        # 这样 run() 可以保持原来的行为：
        # objects=[] -> 跳过增强。
        return img, objects, False

    def random_brightness_contrast(
            self,
            img,
            brightness=0.1,
            contrast=0.15
    ):
        if random.random() > 0.8:
            return img

        img = img.astype(np.float32)

        alpha = 1.0 + random.uniform(
            -contrast,
            contrast
        )

        beta = random.uniform(
            -brightness,
            brightness
        ) * 255

        img = img * alpha + beta

        return np.clip(
            img,
            0,
            255
        ).astype(np.uint8)

    def add_gaussian_noise(
            self,
            img,
            mean=0,
            std=10,
            prob=0.5
    ):
        if random.random() > prob:
            return img

        noise = np.random.normal(
            mean,
            std,
            img.shape
        ).astype(np.float32)

        img = img.astype(np.float32) + noise

        return np.clip(
            img,
            0,
            255
        ).astype(np.uint8)

    def random_hsv(self, img):
        if random.random() > self.hsv_prob:
            return img

        r = (
                np.random.uniform(-1, 1, 3)
                * np.array(
            [
                self.h_gain,
                self.s_gain,
                self.v_gain
            ]
        )
        )

        h, s, v = r

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2HSV
        ).astype(np.float32)

        img[..., 0] = (
                              img[..., 0] + h * 180
                      ) % 180

        img[..., 1] *= (1 + s)
        img[..., 2] *= (1 + v)

        img[..., 1:] = np.clip(
            img[..., 1:],
            0,
            255
        )

        return cv2.cvtColor(
            img.astype(np.uint8),
            cv2.COLOR_HSV2BGR
        )

    def random_flip(self, img, objects):
        if random.random() < self.flip_prob:
            img = np.fliplr(img).copy()

            w = img.shape[1]

            for obj in objects:
                obj["poly"][:, 0] = (
                        w - obj["poly"][:, 0]
                )

        return img, objects

    def cutout_with_mask_raster_safe(
            self,
            img,
            objects,
            max_h_ratio=0.4,
            max_w_ratio=0.4,
            num_holes=(1, 3),
            max_try=50
    ):
        h, w = img.shape[:2]

        if not objects:
            return img, objects

        num_holes = random.randint(*num_holes)

        union_mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        for obj in objects:
            pts = obj["poly"].astype(np.int32)

            cv2.fillPoly(
                union_mask,
                [pts],
                255
            )

        cutouts = []

        for _ in range(num_holes):
            for _try in range(max_try):

                ch = random.randint(
                    max(1, int(0.05 * h)),
                    max(1, int(max_h_ratio * h))
                )

                cw = random.randint(
                    max(1, int(0.05 * w)),
                    max(1, int(max_w_ratio * w))
                )

                if ch >= h or cw >= w:
                    continue

                x = random.randint(
                    0,
                    w - cw
                )

                y = random.randint(
                    0,
                    h - ch
                )

                roi = union_mask[
                      y:y + ch,
                      x:x + cw
                      ]

                # 与原增强代码保持一致：
                # 如果 cutout 完全位于 polygon 内，则放弃。
                if np.all(roi == 255):
                    continue

                cutouts.append(
                    (x, y, cw, ch)
                )

                img[
                y:y + ch,
                x:x + cw
                ] = np.random.randint(
                    0,
                    255,
                    (ch, cw, 3),
                    dtype=np.uint8
                )

                break

        new_objects = []

        for obj in objects:

            obj_mask = np.zeros(
                (h, w),
                dtype=np.uint8
            )

            pts = obj["poly"].astype(np.int32)

            cv2.fillPoly(
                obj_mask,
                [pts],
                255
            )

            for x, y, cw, ch in cutouts:
                obj_mask[
                y:y + ch,
                x:x + cw
                ] = 0

            contours, hierarchy = cv2.findContours(
                obj_mask,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if hierarchy is None:
                continue

            for i, cnt in enumerate(contours):

                # 只保留外轮廓
                if hierarchy[0][i][3] != -1:
                    continue

                if len(cnt) >= 3:
                    new_objects.append({
                        "cls": obj["cls"],
                        "poly": cnt.reshape(-1, 2)
                    })

        return img, new_objects

    def random_rotate(self, img, objects):
        if random.random() > 0.35:
            return img, objects

        h, w = img.shape[:2]

        angle = random.uniform(
            -self.degrees,
            self.degrees
        )

        cx, cy = w / 2, h / 2

        M = cv2.getRotationMatrix2D(
            (cx, cy),
            angle,
            1.0
        )

        cos = abs(M[0, 0])
        sin = abs(M[0, 1])

        new_w = int(
            (h * sin) + (w * cos)
        )

        new_h = int(
            (h * cos) + (w * sin)
        )

        M[0, 2] += (
                           new_w / 2
                   ) - cx

        M[1, 2] += (
                           new_h / 2
                   ) - cy

        rotated_img = cv2.warpAffine(
            img,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderValue=(114, 114, 114)
        )

        new_objects = []

        for obj in objects:

            poly = obj["poly"]

            ones = np.ones(
                (poly.shape[0], 1),
                dtype=np.float32
            )

            pts = np.hstack(
                [poly, ones]
            )

            rotated_pts = pts @ M.T

            if (
                    rotated_pts[:, 0].max() < 0
                    or rotated_pts[:, 1].max() < 0
                    or rotated_pts[:, 0].min() > new_w
                    or rotated_pts[:, 1].min() > new_h
            ):
                continue

            rotated_pts[:, 0] = np.clip(
                rotated_pts[:, 0],
                0,
                new_w - 1
            )

            rotated_pts[:, 1] = np.clip(
                rotated_pts[:, 1],
                0,
                new_h - 1
            )

            if len(rotated_pts) >= 3:
                new_objects.append({
                    "cls": obj["cls"],
                    "poly": rotated_pts
                })

        return rotated_img, new_objects

    def apply_pipeline(self, img, objects):

        # 每次都复制，防止修改原始 objects。
        work_objects = [
            {
                "cls": obj["cls"],
                "poly": obj["poly"].copy()
            }
            for obj in objects
        ]

        work_img = img.copy()

        # 先几何变换
        work_img, work_objects = self.random_rotate(
            work_img,
            work_objects
        )

        # 颜色增强
        work_img = self.random_hsv(
            work_img
        )

        work_img = self.random_brightness_contrast(
            work_img
        )

        work_img = self.add_gaussian_noise(
            work_img,
            mean=0,
            std=10,
            prob=0.5
        )

        # 空间增强
        work_img, work_objects = self.random_flip(
            work_img,
            work_objects
        )

        # Cutout
        work_img, work_objects = (
            self.cutout_with_mask_raster_safe(
                work_img,
                work_objects
            )
        )

        # 最后 Resize （压缩图像大小，防止内存占用较大）
        work_img, work_objects = self.resize_to_max_size(
            work_img,
            work_objects,
            max_size=720
        )

        return work_img, work_objects

    @staticmethod
    def _build_output_name(
            original_filename,
            augment_index
    ):
        """
        命名规则：
            Augmentor_ + 原文件名主体 + _序号

        例如：
            001.jpg
            ->
            Augmentor_001_aug_001.jpg

        这样既满足以 Augmentor_ + 原文件名为核心，
        又避免同一原图生成多次时发生覆盖。
        """

        stem = Path(
            original_filename
        ).stem

        suffix = Path(
            original_filename
        ).suffix.lower()

        return (
            f"Augmentor_{stem}_"
            f"aug_"
            f"{augment_index:03d}"
            f"{suffix}"
        )

    def save_sample(
            self,
            original_filename,
            augment_index,
            img,
            objects
    ):
        h, w = img.shape[:2]

        image_name = self._build_output_name(
            original_filename,
            augment_index
        )

        label_name = (
                Path(image_name).stem
                + ".txt"
        )

        output_image = os.path.join(
            self.output_dir,
            "images",
            image_name
        )

        output_label = os.path.join(
            self.output_dir,
            "labels",
            label_name
        )

        if not cv2.imwrite(
                output_image,
                img
        ):
            raise RuntimeError(
                f"增强图片保存失败：{output_image}"
            )

        # --------------------------------------------------------
        # 注意：
        # 即使 objects == []，
        # 这里仍然会创建 txt 文件。
        #
        # 因此：
        #
        # objects 有内容
        #     -> 写入正常 polygon Label
        #
        # objects == []
        #     -> 创建 0 字节空 Label
        #
        # 这正好符合当前需求。
        # --------------------------------------------------------
        with open(
                output_label,
                "w",
                encoding="utf-8"
        ) as f:

            for obj in objects:

                poly = obj["poly"].astype(
                    np.float32
                )

                poly[:, 0] /= w
                poly[:, 1] /= h

                poly = np.clip(
                    poly,
                    0,
                    1
                ).reshape(-1)

                if len(poly) < 6:
                    continue

                line = " ".join(
                    [str(obj["cls"])]
                    + [
                        f"{v:.6f}"
                        for v in poly
                    ]
                )

                f.write(line + "\n")

    def run(self):
        """
        对 exist_target_dataset 中的每张图片，
        按 augment_per_image 生成增强样本。

        空 Label 图片：
            仍然进行图像增强，
            并生成对应的空 Label。
        """

        total_expected = (
                len(self.img_files)
                * self.augment_per_image
        )

        print("\n" + "=" * 80)
        print("开始 Hard Case 样本增强")
        print("=" * 80)

        with tqdm(
                total=total_expected,
                desc="Hard Case Augmenting",
                unit="img",
                ncols=100,
                dynamic_ncols=True
        ) as pbar:

            for img_path in self.img_files:

                original_filename = os.path.basename(
                    img_path
                )

                try:

                    img, objects, label_is_empty = (
                        self.load_image_and_labels(
                            img_path
                        )
                    )

                    # ====================================================
                    # 关键修改：
                    #
                    # 只有“真正的空 Label 文件”才允许 objects=[] 继续增强。
                    #
                    # 非空 Label 但解析不到有效 polygon，
                    # 仍保持原来的行为：跳过。
                    # ====================================================

                    if not objects and not label_is_empty:
                        print(
                            f"[WARNING] Label无有效Polygon，"
                            f"跳过增强："
                            f"{original_filename}"
                        )

                        self.failed_count += (
                            self.augment_per_image
                        )

                        pbar.update(
                            self.augment_per_image
                        )

                        continue

                    # ====================================================
                    # 空 Label：
                    #
                    # objects = []
                    # label_is_empty = True
                    #
                    # 不再跳过。
                    # ====================================================

                    if label_is_empty:
                        print(
                            f"[INFO] Label为空，"
                            f"仍执行增强："
                            f"{original_filename}"
                        )

                    for augment_index in range(
                            1,
                            self.augment_per_image + 1
                    ):

                        # 避免极端随机增强后 polygon 全部消失。
                        #
                        # 对于空 Label：
                        #     objects 本来就是 []
                        #     因此不需要“目标全部消失”的判断。
                        #
                        success = False

                        for _try in range(10):

                            aug_img, aug_objects = (
                                self.apply_pipeline(
                                    img,
                                    objects
                                )
                            )

                            # ------------------------------------------------
                            # 空 Label 是合法样本。
                            #
                            # 如果原始 Label 为空，
                            # 那么 aug_objects == [] 是正常结果。
                            #
                            # 因此直接认为增强成功。
                            # ------------------------------------------------
                            if label_is_empty:
                                success = True
                                break

                            # ------------------------------------------------
                            # 原有逻辑保持不变：
                            # 有目标 Label 如果增强后目标全部消失，
                            # 则重新生成。
                            # ------------------------------------------------
                            if aug_objects:
                                success = True
                                break

                        if not success:
                            print(
                                f"[WARNING] 增强失败，"
                                f"目标全部消失："
                                f"{original_filename}"
                            )

                            self.failed_count += 1

                            pbar.update(1)

                            continue

                        self.save_sample(
                            original_filename,
                            augment_index,
                            aug_img,
                            aug_objects
                        )

                        self.success_count += 1

                        pbar.update(1)

                except Exception as e:

                    print(
                        f"[ERROR] 增强失败："
                        f"{original_filename} | {e}"
                    )

                    self.failed_count += (
                        self.augment_per_image
                    )

                    pbar.update(
                        self.augment_per_image
                    )

                pbar.set_postfix({
                    "success": self.success_count,
                    "failed": self.failed_count
                })

        print("\n" + "=" * 80)
        print("Hard Case 样本增强完成")
        print("=" * 80)
        print(f"原始样本数量：      {len(self.img_files)}")
        print(f"计划生成数量：      {total_expected}")
        print(f"成功生成数量：      {self.success_count}")
        print(f"失败数量：          {self.failed_count}")
        print(f"输出目录：          {self.output_dir}")
        print("=" * 80)


def augment_exist_target_dataset(
        target_dataset: str,
        augment_ratio: float = 2.0
):
    """
    对：
        TARGET_DATASET/exist_target_dataset

    进行增强，并输出到：
        TARGET_DATASET/Augmentor_exist_target_dataset
    """

    exist_images_dir = os.path.join(
        target_dataset,
        "exist_target_dataset",
        "images"
    )

    exist_labels_dir = os.path.join(
        target_dataset,
        "exist_target_dataset",
        "labels"
    )

    augment_output_dir = os.path.join(
        target_dataset,
        "Augmentor_exist_target_dataset"
    )

    if not os.path.isdir(exist_images_dir):
        raise FileNotFoundError(
            f"找不到 exist_target_dataset/images："
            f"{exist_images_dir}"
        )

    if not os.path.isdir(exist_labels_dir):
        raise FileNotFoundError(
            f"找不到 exist_target_dataset/labels："
            f"{exist_labels_dir}"
        )

    augmentor = YOLOSegAugmentor(
        img_dir=exist_images_dir,
        label_dir=exist_labels_dir,
        output_dir=augment_output_dir,
        augment_ratio=augment_ratio
    )

    augmentor.run()

    # 增强数据集自身做一次 images / labels 对应性检查。
    check_dataset_pair(
        os.path.join(
            augment_output_dir,
            "images"
        ),
        os.path.join(
            augment_output_dir,
            "labels"
        ),
        "Augmentor_exist_target_dataset"
    )

    return augment_output_dir


def cleanup_dataset_pair(images_dir, labels_dir, dataset_name):
    """
    清理 images / labels 中无法一一对应的孤儿文件。

    规则：

        image 有、label 无
            -> 删除 image

        label 有、image 无
            -> 删除 label
    """

    print("\n" + "=" * 80)
    print(f"开始清理 {dataset_name} 孤儿文件")
    print("=" * 80)

    image_map = {}
    label_map = {}

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    if os.path.isdir(images_dir):

        for filename in os.listdir(images_dir):

            ext = os.path.splitext(filename)[1].lower()

            if ext not in IMAGE_EXTENSIONS:
                continue

            stem = os.path.splitext(filename)[0]

            image_map[stem] = os.path.join(
                images_dir,
                filename
            )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    if os.path.isdir(labels_dir):

        for filename in os.listdir(labels_dir):

            if not filename.lower().endswith(".txt"):
                continue

            stem = os.path.splitext(filename)[0]

            label_map[stem] = os.path.join(
                labels_dir,
                filename
            )

    # --------------------------------------------------------
    # Image 有，Label 没有
    # --------------------------------------------------------

    orphan_images = set(image_map) - set(label_map)

    # --------------------------------------------------------
    # Label 有，Image 没有
    # --------------------------------------------------------

    orphan_labels = set(label_map) - set(image_map)

    # --------------------------------------------------------
    # 删除孤儿 Image
    # --------------------------------------------------------

    for stem in sorted(orphan_images):
        path = image_map[stem]

        print(
            f"[DELETE] Image 无对应 Label：{path}"
        )

        os.remove(path)

    # --------------------------------------------------------
    # 删除孤儿 Label
    # --------------------------------------------------------

    for stem in sorted(orphan_labels):
        path = label_map[stem]

        print(
            f"[DELETE] Label 无对应 Image：{path}"
        )

        os.remove(path)

    # --------------------------------------------------------
    # 输出
    # --------------------------------------------------------

    print(
        f"删除孤儿 Image：{len(orphan_images)}"
    )

    print(
        f"删除孤儿 Label：{len(orphan_labels)}"
    )

    if not orphan_images and not orphan_labels:
        print("✓ 没有发现孤儿文件")

    print("=" * 80)


# ============================================================
# 根据 JSON 创建目标数据集
# ============================================================


def create_target_datasets(
        local_save_dir: str,
        hard_case_info_json: str,
        TARGET_DATASET,
        random_seed=42,
        unexitst_sample_ratio=1 / 3,
):
    """
    根据 hard_case_info.json 创建：

        exist_target_dataset/
        ├── images/
        └── labels/

        unexist_target_dataset/
        ├── images/
        └── labels/


    exist_target_dataset：

        将 labeled_images 中的图片
        拷贝到：

            exist_target_dataset/images/

        将 labels 中的 txt
        拷贝到：

            exist_target_dataset/labels/


    unexist_target_dataset：

        从 unlabeled_images 中随机抽取 1/3。

        图片：

            unexist_target_dataset/images/

        同时创建空 label：

            unexist_target_dataset/labels/


    保证 images / labels basename 一一对应。
    """

    print("\n" + "=" * 80)
    print("开始生成目标数据集")
    print("=" * 80)

    # ========================================================
    # 检查 JSON
    # ========================================================

    if not os.path.isfile(hard_case_info_json):
        raise FileNotFoundError(f"找不到 hard_case_info.json：" f"{hard_case_info_json}")

    # ========================================================
    # 读取 JSON
    # ========================================================

    print(f"读取 JSON：{hard_case_info_json}")

    with open(hard_case_info_json, "r", encoding="utf-8") as f:

        hard_case_info = json.load(f)

    labeled_images = hard_case_info.get("labeled_images", [])

    unlabeled_images = hard_case_info.get("unlabeled_images", [])

    labels = hard_case_info.get("labels", [])

    print(f"JSON中的有标注图片：" f"{len(labeled_images)}")

    print(f"JSON中的无标注图片：" f"{len(unlabeled_images)}")

    print(f"JSON中的Label：" f"{len(labels)}")

    # ========================================================
    # 创建目录
    # ========================================================

    exist_images_dir = os.path.join(TARGET_DATASET, "exist_target_dataset", "images")

    exist_labels_dir = os.path.join(TARGET_DATASET, "exist_target_dataset", "labels")

    unexist_images_dir = os.path.join(
        TARGET_DATASET, "unexist_target_dataset", "images"
    )

    unexist_labels_dir = os.path.join(
        TARGET_DATASET, "unexist_target_dataset", "labels"
    )

    os.makedirs(exist_images_dir, exist_ok=True)

    os.makedirs(exist_labels_dir, exist_ok=True)

    os.makedirs(unexist_images_dir, exist_ok=True)

    os.makedirs(unexist_labels_dir, exist_ok=True)

    print("\n目标目录：")

    print(f"有标注图片：" f"{exist_images_dir}")

    print(f"有标注Label：" f"{exist_labels_dir}")

    print(f"无标注图片：" f"{unexist_images_dir}")

    print(f"无标注Label：" f"{unexist_labels_dir}")

    # ========================================================
    # 建立 Label basename 映射
    #
    # 用于确保：
    #
    # xxx.jpg
    #
    # 对应：
    #
    # xxx.txt
    #
    # ========================================================

    label_map = {}

    for label_path in labels:
        filename = os.path.basename(label_path)

        stem = os.path.splitext(filename)[0]

        label_map[stem] = label_path

    # ========================================================
    # 处理有标注数据
    # ========================================================

    print("\n" + "-" * 80)
    print("开始复制有标注数据")
    print("-" * 80)

    exist_success_count = 0
    exist_failed_count = 0

    for image_relative_path in labeled_images:

        # ----------------------------------------------------
        # 获取原始图片路径
        # ----------------------------------------------------

        source_image = os.path.join(local_save_dir, image_relative_path)

        if not os.path.isfile(source_image):
            print(f"[ERROR] 图片不存在：" f"{source_image}")

            exist_failed_count += 1

            continue

        image_filename = os.path.basename(image_relative_path)

        image_stem = os.path.splitext(image_filename)[0]

        # ----------------------------------------------------
        # 查找对应 Label
        # ----------------------------------------------------

        if image_stem not in label_map:
            print(f"[ERROR] 找不到对应 Label：" f"{image_relative_path}")

            exist_failed_count += 1

            continue

        label_relative_path = label_map[image_stem]

        source_label = os.path.join(local_save_dir, label_relative_path)

        if not os.path.isfile(source_label):
            print(f"[ERROR] Label不存在：" f"{source_label}")

            exist_failed_count += 1

            continue

        # ----------------------------------------------------
        # 目标路径
        # ----------------------------------------------------

        target_image = os.path.join(exist_images_dir, image_filename)

        target_label = os.path.join(exist_labels_dir, image_stem + ".txt")

        # ----------------------------------------------------
        # 检查目标文件名冲突
        # ----------------------------------------------------

        if os.path.exists(target_image):
            print(f"[WARNING] 目标图片已存在，跳过：" f"{target_image}")

            continue

        if os.path.exists(target_label):
            print(f"[WARNING] 目标Label已存在，跳过：" f"{target_label}")

            continue

        # ----------------------------------------------------
        # 拷贝图片
        # ----------------------------------------------------

        shutil.copy2(source_image, target_image)

        # ----------------------------------------------------
        # 拷贝 Label
        # ----------------------------------------------------

        shutil.copy2(source_label, target_label)

        exist_success_count += 1

    # ========================================================
    # 输出有标注数据结果
    # ========================================================

    print(f"\n有标注数据复制完成：")

    print(f"成功：{exist_success_count}")

    print(f"失败：{exist_failed_count}")

    # ========================================================
    # 随机抽取 1/3 无标注图片
    # ========================================================

    print("\n" + "-" * 80)
    print("开始随机抽取无标注图片")
    print("-" * 80)

    total_unlabeled = len(unlabeled_images)

    # ========================================================
    # 计算 1/3 数量
    #
    # 使用向下取整。
    #
    # 例如：
    #
    # 10 -> 3
    # 11 -> 3
    # 12 -> 4
    #
    # ========================================================

    sample_count = int(total_unlabeled * unexitst_sample_ratio)

    print(f"无标注图片总数：" f"{total_unlabeled}")

    print(f"随机抽取数量：" f"{sample_count}")

    # ========================================================
    # 设置随机种子
    # ========================================================

    if random_seed is not None:
        random.seed(random_seed)

    # ========================================================
    # 随机抽样
    # ========================================================

    selected_unlabeled_images = random.sample(unlabeled_images, sample_count)

    # ========================================================
    # 排序
    #
    # 抽样本身是随机的，
    # 但保存前排序可以让最终文件处理顺序稳定。
    #
    # ========================================================

    selected_unlabeled_images.sort()

    # ========================================================
    # 拷贝无标注图片 + 创建空 Label
    # ========================================================

    unexist_success_count = 0
    unexist_failed_count = 0

    for image_relative_path in selected_unlabeled_images:

        # ----------------------------------------------------
        # 原始图片
        # ----------------------------------------------------

        source_image = os.path.join(local_save_dir, image_relative_path)

        if not os.path.isfile(source_image):
            print(f"[ERROR] 图片不存在：" f"{source_image}")

            unexist_failed_count += 1

            continue

        image_filename = os.path.basename(image_relative_path)

        image_stem = os.path.splitext(image_filename)[0]

        # ----------------------------------------------------
        # 目标图片
        # ----------------------------------------------------

        target_image = os.path.join(unexist_images_dir, image_filename)

        target_label = os.path.join(unexist_labels_dir, image_stem + ".txt")

        # ----------------------------------------------------
        # 检查目标冲突
        # ----------------------------------------------------

        if os.path.exists(target_image):
            print(f"[WARNING] 目标图片已存在：" f"{target_image}")

            unexist_failed_count += 1

            continue

        if os.path.exists(target_label):
            print(f"[WARNING] 目标Label已存在：" f"{target_label}")

            unexist_failed_count += 1

            continue

        # ----------------------------------------------------
        # 拷贝图片
        # ----------------------------------------------------

        shutil.copy2(source_image, target_image)

        # ----------------------------------------------------
        # 创建空 Label
        # ----------------------------------------------------

        with open(target_label, "w", encoding="utf-8") as f:

            pass

        unexist_success_count += 1

    # ========================================================
    # 清理孤儿文件
    # ========================================================
    # Image 有、Label 无
    #     -> 删除 Image
    #
    # Label 有、Image 无
    #     -> 删除 Label
    #
    # 清理完成后，再进行最终对应关系检查。
    # ========================================================

    print("\n" + "-" * 80)
    print("开始清理 images / labels 孤儿文件")
    print("-" * 80)

    cleanup_dataset_pair(
        exist_images_dir,
        exist_labels_dir,
        "exist_target_dataset"
    )

    cleanup_dataset_pair(
        unexist_images_dir,
        unexist_labels_dir,
        "unexist_target_dataset"
    )

    # ========================================================
    # 最终检查 images / labels 是否一一对应
    # ========================================================

    print("\n" + "-" * 80)
    print("开始检查 images / labels 对应关系")
    print("-" * 80)

    check_dataset_pair(exist_images_dir, exist_labels_dir, "exist_target_dataset")

    check_dataset_pair(unexist_images_dir, unexist_labels_dir, "unexist_target_dataset")

    # ========================================================
    # 最终输出
    # ========================================================

    print("\n" + "=" * 80)
    print("目标数据集生成完成")
    print("=" * 80)

    print(f"有标注数据：" f"{exist_success_count}")

    print(f"有标注失败：" f"{exist_failed_count}")

    print(f"无标注抽样数量：" f"{sample_count}")

    print(f"无标注成功：" f"{unexist_success_count}")

    print(f"无标注失败：" f"{unexist_failed_count}")

    print("=" * 80)


# ============================================================
# 检查 images / labels 是否一一对应
# ============================================================


def check_dataset_pair(images_dir: str, labels_dir: str, dataset_name: str):
    """
    检查：

        images/xxx.jpg
        labels/xxx.txt

    是否一一对应。
    """

    image_stems = set()

    label_stems = set()

    # ========================================================
    # 图片
    # ========================================================

    if os.path.isdir(images_dir):

        for filename in os.listdir(images_dir):

            extension = os.path.splitext(filename)[1].lower()

            if extension not in IMAGE_EXTENSIONS:
                continue

            stem = os.path.splitext(filename)[0]

            image_stems.add(stem)

    # ========================================================
    # Label
    # ========================================================

    if os.path.isdir(labels_dir):

        for filename in os.listdir(labels_dir):

            if not filename.lower().endswith(".txt"):
                continue

            stem = os.path.splitext(filename)[0]

            label_stems.add(stem)

    # ========================================================
    # 找差异
    # ========================================================

    missing_labels = image_stems - label_stems

    missing_images = label_stems - image_stems

    print(f"\n[{dataset_name}]")

    print(f"Images：" f"{len(image_stems)}")

    print(f"Labels：" f"{len(label_stems)}")

    print(f"缺少 Label：" f"{len(missing_labels)}")

    print(f"缺少 Image：" f"{len(missing_images)}")

    if missing_labels:

        print("[ERROR] 以下图片没有对应 Label：")

        for stem in sorted(missing_labels)[:20]:
            print(f"    {stem}")

    if missing_images:

        print("[ERROR] 以下 Label 没有对应 Image：")

        for stem in sorted(missing_images)[:20]:
            print(f"    {stem}")

    if not missing_labels and not missing_images:
        print("✓ Images / Labels 一一对应")


def generate_image_path_txt(hard_case_exist_target_images_dir, base_prefix):
    """
    读取指定文件夹内的所有图片名称，拼接基础路径字符串后写入 txt 文件。

    :param image_folder: 存放图片的文件夹路径
    :param base_path_str: 要拼接的前缀路径字符串
    :param output_txt_path: 导出的 txt 文件保存路径
    """

    image_folder = os.path.join(
        hard_case_exist_target_images_dir, "exist_target_dataset", "images"
    )
    base_path_str = base_prefix
    output_txt_path = os.path.join(
        hard_case_exist_target_images_dir, "exist_target_dataset", "hard_cases.txt"
    )

    # 常见的图像文件扩展名集合
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif")

    # 获取文件夹下所有指定格式的图像文件名
    image_names = [
        f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)
    ]

    # 按照文件名排序（可选，保持文件列表整洁）
    image_names.sort()

    # 写入 txt 文件
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for img_name in image_names:
            # 自动处理路径分隔符，将前缀与图片文件名拼接
            new_path = os.path.join(base_path_str, img_name)
            f.write(new_path + "\n")

    print(f"成功处理 {len(image_names)} 张图片，结果已保存至: {output_txt_path}")


# ============================================================
# 创建数据集增量信息记录
# ============================================================

def update_dataset_increment_info(
        target_dataset: str
):
    """
    在 TARGET_DATASET 的上一层目录下，
    创建 / 更新：

        dataset_increment_info.json

    例如：

        TARGET_DATASET =
        /data/database/AITotal_SegmentDatabase/
        carpetDatabaseSegment/date_20260915

    则：

        /data/database/AITotal_SegmentDatabase/
        carpetDatabaseSegment/dataset_increment_info.json


    JSON 结构：

    {
        "date_20260915": {
            "batch_id": "date_20260915",
            "processed_time": "2026-09-16 10:25:30",
            "exist_target_image_count": 856,
            "unexist_target_image_count": 126,
            "total_image_count": 982
        },

        "date_20260916": {
            ...
        }
    }


    参数：
        target_dataset:
            最终目标数据集目录
    """

    print("\n" + "=" * 80)
    print("开始更新数据集增量信息")
    print("=" * 80)

    # ========================================================
    # 检查 TARGET_DATASET
    # ========================================================

    if not os.path.isdir(target_dataset):
        raise FileNotFoundError(
            f"TARGET_DATASET 不存在：{target_dataset}"
        )

    # ========================================================
    # 获取 TARGET_DATASET 最终目录名
    #
    # 例如：
    #
    # /xxx/carpetDatabaseSegment/date_20260915
    #
    # 得到：
    #
    # date_20260915
    # ========================================================

    target_dataset_name = os.path.basename(
        os.path.normpath(target_dataset)
    )

    # ========================================================
    # TARGET_DATASET 上一级目录
    #
    # 例如：
    #
    # /xxx/carpetDatabaseSegment/date_20260915
    #
    # 上一级：
    #
    # /xxx/carpetDatabaseSegment
    # ========================================================

    parent_dir = os.path.dirname(
        os.path.normpath(target_dataset)
    )

    # ========================================================
    # 增量信息 JSON
    # ========================================================

    increment_info_path = os.path.join(
        parent_dir,
        "dataset_increment_info.json"
    )

    # ========================================================
    # exist_target_dataset
    # ========================================================

    exist_images_dir = os.path.join(
        target_dataset,
        "exist_target_dataset",
        "images"
    )

    # ========================================================
    # unexist_target_dataset
    # ========================================================

    unexist_images_dir = os.path.join(
        target_dataset,
        "unexist_target_dataset",
        "images"
    )

    # ========================================================
    # 检查目录
    # ========================================================

    if not os.path.isdir(exist_images_dir):
        raise FileNotFoundError(
            "找不到 exist_target_dataset/images："
            f"{exist_images_dir}"
        )

    if not os.path.isdir(unexist_images_dir):
        raise FileNotFoundError(
            "找不到 unexist_target_dataset/images："
            f"{unexist_images_dir}"
        )

    # ========================================================
    # 统计 exist_target_dataset/images
    # ========================================================

    exist_image_count = 0

    for filename in os.listdir(
            exist_images_dir
    ):

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension in IMAGE_EXTENSIONS:
            exist_image_count += 1

    # ========================================================
    # 统计 unexist_target_dataset/images
    # ========================================================

    unexist_image_count = 0

    for filename in os.listdir(
            unexist_images_dir
    ):

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension in IMAGE_EXTENSIONS:
            unexist_image_count += 1

    # ========================================================
    # 总图片数量
    # ========================================================

    total_image_count = (
            exist_image_count
            +
            unexist_image_count
    )

    # ========================================================
    # 当前处理时间
    #
    # 使用本地时间。
    # ========================================================

    from datetime import datetime

    processed_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # ========================================================
    # 如果增量信息 JSON 已经存在，则读取
    #
    # 这样不会覆盖之前的数据批次。
    # ========================================================

    if os.path.isfile(
            increment_info_path
    ):

        try:

            with open(
                    increment_info_path,
                    "r",
                    encoding="utf-8"
            ) as f:

                increment_info = json.load(f)

            if not isinstance(
                    increment_info,
                    dict
            ):
                print(
                    "[WARNING] 增量信息文件格式异常，"
                    "重新创建。"
                )

                increment_info = {}

        except Exception as e:

            print(
                "[WARNING] 读取已有增量信息失败："
                f"{e}"
            )

            print(
                "将重新创建增量信息文件。"
            )

            increment_info = {}

    else:

        increment_info = {}

    # ========================================================
    # 当前数据批次
    # ========================================================

    current_batch_info = {

        # 数据批次 ID
        "batch_id":
            target_dataset_name,

        # 最后处理时间
        "processed_time":
            processed_time,

        # 有标注目标数据集图片数量
        "exist_target_image_count":
            exist_image_count,

        # 无标注目标数据集图片数量
        "unexist_target_image_count":
            unexist_image_count,

        # 总图片数量
        "total_image_count":
            total_image_count,
    }

    # ========================================================
    # 写入 / 更新当前批次
    #
    # 以 TARGET_DATASET 最终目录名作为 key
    # ========================================================

    increment_info[
        target_dataset_name
    ] = current_batch_info

    # ========================================================
    # 排序
    #
    # 让 JSON 中的 batch 按目录名排序，
    # 方便后续查看。
    # ========================================================

    increment_info = dict(
        sorted(
            increment_info.items()
        )
    )

    # ========================================================
    # 写入 JSON
    # ========================================================

    with open(
            increment_info_path,
            "w",
            encoding="utf-8"
    ) as f:

        json.dump(
            increment_info,
            f,
            ensure_ascii=False,
            indent=4
        )

    # ========================================================
    # 输出结果
    # ========================================================

    print(
        f"数据批次：              "
        f"{target_dataset_name}"
    )

    print(
        f"最后处理时间：          "
        f"{processed_time}"
    )

    print(
        f"exist_target 图片数量： "
        f"{exist_image_count}"
    )

    print(
        f"unexist_target 图片数量："
        f"{unexist_image_count}"
    )

    print(
        f"总图片数量：             "
        f"{total_image_count}"
    )

    print(
        f"增量信息文件：           "
        f"{increment_info_path}"
    )

    print("=" * 80)


# ============================================================
# 增量更新 total_hard_cases.txt
# ============================================================

def update_total_hard_cases(target_dataset: str):
    """
    将当前批次生成的 hard_cases.txt 增量合并到：

        LOCAL_SAVE_DIR 的上一级目录 / total_hard_cases.txt

    处理规则：

    1. 读取当前批次 hard_cases.txt
    2. 读取已有 total_hard_cases.txt
    3. 历史记录保持原有顺序
    4. 当前批次中已经存在于 total_hard_cases.txt 的记录不再追加
    5. 当前批次内部重复记录只保留一次
    6. 新记录按照当前 hard_cases.txt 的顺序追加到末尾
    7. 最终 total_hard_cases.txt 不存在重复记录
    """

    print("\n" + "=" * 80)
    print("开始更新 total_hard_cases.txt")
    print("=" * 80)

    # ========================================================
    # 当前批次 hard_cases.txt
    # ========================================================

    current_hard_cases_path = os.path.join(
        target_dataset,
        "exist_target_dataset",
        "hard_cases.txt"
    )

    if not os.path.isfile(current_hard_cases_path):
        raise FileNotFoundError(
            f"找不到当前批次 hard_cases.txt："
            f"{current_hard_cases_path}"
        )

    # ========================================================
    # LOCAL_SAVE_DIR 上一级目录
    # ========================================================

    parent_dir = os.path.dirname(
        os.path.normpath(target_dataset)
    )

    total_hard_cases_path = os.path.join(
        parent_dir,
        "total_hard_cases.txt"
    )

    # ========================================================
    # 读取当前批次 hard_cases.txt
    # ========================================================

    current_records = []

    with open(
            current_hard_cases_path,
            "r",
            encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            current_records.append(line)

    # ========================================================
    # 当前批次去重
    #
    # 保持原始顺序
    # ========================================================

    current_unique_records = []
    current_seen = set()

    for record in current_records:

        if record in current_seen:
            continue

        current_seen.add(record)
        current_unique_records.append(record)

    # ========================================================
    # 读取历史 total_hard_cases.txt
    # ========================================================

    existing_records = []
    existing_seen = set()

    if os.path.isfile(total_hard_cases_path):

        with open(
                total_hard_cases_path,
                "r",
                encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                # 历史文件如果存在重复，
                # 这里也直接去掉
                if line in existing_seen:
                    continue

                existing_seen.add(line)
                existing_records.append(line)

    # ========================================================
    # 找出当前批次真正新增的记录
    # ========================================================

    new_records = []

    for record in current_unique_records:

        if record in existing_seen:
            continue

        existing_seen.add(record)
        new_records.append(record)

    # ========================================================
    # 合并最终结果
    #
    # 历史记录保持原顺序
    # 新记录追加到最后
    # ========================================================

    all_records = existing_records + new_records

    # ========================================================
    # 写回 total_hard_cases.txt
    # ========================================================

    with open(
            total_hard_cases_path,
            "w",
            encoding="utf-8"
    ) as f:

        for record in all_records:
            f.write(record + "\n")

    # ========================================================
    # 输出统计
    # ========================================================

    print(f"当前批次原始记录数量：       {len(current_records)}")
    print(f"当前批次去重后数量：         {len(current_unique_records)}")
    print(f"历史已有记录数量：           {len(existing_records)}")
    print(f"本次新增记录数量：           {len(new_records)}")
    print(f"累计 Hard Case 数量：        {len(all_records)}")
    print(f"累计文件：                   {total_hard_cases_path}")

    print("=" * 80)


# ============================================================
# Main
# ============================================================
# ============================================================
# 配置
# ============================================================

LOCAL_SAVE_DIR = (
    "/data/database/AITotal_SegmentDatabase/"
    "cvat_dataset/carpet_detect/date_carpet_20260915"
)

# ============================================================
# JSON 文件
# ============================================================

HARD_CASE_INFO_JSON = os.path.join(LOCAL_SAVE_DIR, "hard_case_info.json")

# ============================================================
# 输出数据集目录
# ============================================================

TARGET_DATASET = (
    "/data/database/AITotal_SegmentDatabase/carpetDatabaseSegment/date_20260915"
)

# ============================================================
# 随机种子
#
# 设置为固定值可以保证每次运行抽取结果一致。
#
# 如果希望每次运行随机抽取不同的 1/3，
# 可以设置为：
#
# RANDOM_SEED = None
#
# ============================================================

RANDOM_SEED = 42

unexitst_sample_ratio = 1 / 2  # 无标注图片抽取比例

# ============================================================
# Hard Case 样本增强配置
# ============================================================
# 每张 exist_target_dataset 原图生成 3 张增强图。
# 例如：001.jpg -> Augmentor_001_001.jpg / _002.jpg / _003.jpg
AUGMENT_RATIO = 2.0

# 2. 需要拼接的前缀路径字符串 (例如服务器/远程/相对路径字符串)
base_prefix = ("/workspace/data/AITotal_SegmentDatabase/carpetDatabaseSegment/images/train")

if __name__ == "__main__":
    # ========================================================
    # 第一步：
    # 扫描所有 Hard Case 数据集，
    # 生成 hard_case_info.json
    # ========================================================

    generate_hard_case_info(LOCAL_SAVE_DIR)

    # ========================================================
    # 第二步：
    # 根据 hard_case_info.json
    # 创建目标数据集
    # ========================================================

    create_target_datasets(
        LOCAL_SAVE_DIR,
        HARD_CASE_INFO_JSON,
        TARGET_DATASET=TARGET_DATASET,
        random_seed=RANDOM_SEED,
        unexitst_sample_ratio=unexitst_sample_ratio,
    )

    # ========================================================
    # 第三步：
    # 根据 TARGET_DATASET
    # 创建目标数据集的hard case txt文件
    # ========================================================

    # 执行函数
    generate_image_path_txt(TARGET_DATASET, base_prefix)
    # ========================================================
    # 第四步：
    # 更新数据集增量信息
    # ========================================================
    update_dataset_increment_info(TARGET_DATASET)

    # ========================================================
    # 第五步：
    # 增量更新 total_hard_cases.txt
    # ========================================================

    update_total_hard_cases(TARGET_DATASET)

    # # ========================================================
    # # 第六步：增强 exist_target_dataset 中的 Hard Case 样本
    # # ========================================================
    augment_exist_target_dataset(
        TARGET_DATASET,
        augment_ratio=AUGMENT_RATIO,
    )
