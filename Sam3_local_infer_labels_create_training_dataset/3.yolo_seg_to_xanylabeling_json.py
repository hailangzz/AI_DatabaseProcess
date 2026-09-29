#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO-Seg -> X-AnyLabeling JSON

功能：
1. 读取 YOLO-Seg txt 标注
2. 将归一化 polygon 坐标转换为像素坐标
3. 转换成 X-AnyLabeling / LabelMe 风格 JSON
4. 每张图片生成一个同名 JSON
5. 支持一张图片多个目标
6. 支持多个 polygon
7. 无标签图片也可以生成空 JSON

YOLO-Seg 格式：
class_id x1 y1 x2 y2 x3 y3 ...

classes.txt：
carpet
wire
liquid
plasticbag
"""

import argparse
import json
import os

from PIL import Image

# ============================================================
# 配置
# ============================================================

DEFAULT_CLASSES = [
    "carpet",
    "wire",
    "liquid",
    "plasticbag",
]


# ============================================================
# 读取 classes.txt
# ============================================================

def load_classes(classes_path):
    with open(classes_path, "r", encoding="utf-8") as f:
        classes = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return classes


# ============================================================
# YOLO-Seg -> X-AnyLabeling JSON
# ============================================================

def convert_one(
        image_path,
        label_path,
        output_json_path,
        classes,
):
    # --------------------------------------------------------
    # 获取图片尺寸
    # --------------------------------------------------------
    with Image.open(image_path) as img:
        image_width, image_height = img.size

    shapes = []

    # --------------------------------------------------------
    # 没有 label 文件
    # --------------------------------------------------------
    if not os.path.exists(label_path):
        print(f"[WARNING] Label 不存在，生成空 JSON：{label_path}")
    else:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # ----------------------------------------------------
        # 读取每一条 YOLO-Seg 标注
        # ----------------------------------------------------
        for line_num, line in enumerate(lines, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            # 至少：
            # class_id + x1 y1 x2 y2 x3 y3
            if len(parts) < 7:
                print(
                    f"[WARNING] 跳过非法标注："
                    f"{label_path}:{line_num}"
                )
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                print(
                    f"[WARNING] class_id 非法："
                    f"{label_path}:{line_num}"
                )
                continue

            # ------------------------------------------------
            # 类别检查
            # ------------------------------------------------
            if class_id < 0 or class_id >= len(classes):
                print(
                    f"[WARNING] class_id 超出范围："
                    f"{class_id}，文件：{label_path}"
                )
                continue

            label = classes[class_id]

            # ------------------------------------------------
            # polygon 坐标
            # ------------------------------------------------
            coordinate_values = parts[1:]

            # 必须是偶数
            if len(coordinate_values) % 2 != 0:
                print(
                    f"[WARNING] polygon 坐标数量不是偶数："
                    f"{label_path}:{line_num}"
                )
                continue

            points = []

            for i in range(0, len(coordinate_values), 2):

                try:
                    x = float(coordinate_values[i])
                    y = float(coordinate_values[i + 1])
                except ValueError:
                    print(
                        f"[WARNING] 坐标非法："
                        f"{label_path}:{line_num}"
                    )
                    points = []
                    break

                # YOLO normalized -> pixel
                pixel_x = x * image_width
                pixel_y = y * image_height

                points.append([
                    pixel_x,
                    pixel_y
                ])

            if len(points) < 3:
                print(
                    f"[WARNING] polygon 点数量不足，跳过："
                    f"{label_path}:{line_num}"
                )
                continue

            # ------------------------------------------------
            # X-AnyLabeling / LabelMe shape
            # ------------------------------------------------
            shape = {
                "label": label,
                "points": points,
                "group_id": None,
                "description": None,
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            }

            shapes.append(shape)

    # ========================================================
    # 构造 JSON
    # ========================================================

    image_filename = os.path.basename(image_path)

    data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width,
    }

    # --------------------------------------------------------
    # 创建输出目录
    # --------------------------------------------------------
    os.makedirs(
        os.path.dirname(output_json_path),
        exist_ok=True
    )

    # --------------------------------------------------------
    # 保存 JSON
    # --------------------------------------------------------
    with open(
            output_json_path,
            "w",
            encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# 批量转换
# ============================================================

def convert_dataset(
        images_dir,
        labels_dir,
        output_dir,
        classes_path,
):
    classes = load_classes(classes_path)

    print("=" * 70)
    print("YOLO-Seg -> X-AnyLabeling")
    print("=" * 70)

    print(f"Images : {images_dir}")
    print(f"Labels : {labels_dir}")
    print(f"Output : {output_dir}")
    print(f"Classes: {classes}")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # 支持的图片格式
    # --------------------------------------------------------
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    total = 0
    success = 0

    # --------------------------------------------------------
    # 遍历图片
    # --------------------------------------------------------
    for filename in sorted(os.listdir(images_dir)):

        image_path = os.path.join(
            images_dir,
            filename
        )

        if not os.path.isfile(image_path):
            continue

        ext = os.path.splitext(filename)[1].lower()

        if ext not in image_extensions:
            continue

        total += 1

        # ----------------------------------------------------
        # 对应 label
        # ----------------------------------------------------
        basename = os.path.splitext(filename)[0]

        label_path = os.path.join(
            labels_dir,
            basename + ".txt"
        )

        # ----------------------------------------------------
        # 输出 JSON
        # ----------------------------------------------------
        output_json_path = os.path.join(
            output_dir,
            basename + ".json"
        )

        try:

            convert_one(
                image_path=image_path,
                label_path=label_path,
                output_json_path=output_json_path,
                classes=classes,
            )

            success += 1

        except Exception as e:

            print(
                f"[ERROR] 转换失败：{filename}"
            )
            print(
                f"        {type(e).__name__}: {e}"
            )

    print("=" * 70)
    print(f"图片总数 : {total}")
    print(f"转换成功 : {success}")
    print(f"转换失败 : {total - success}")
    print("=" * 70)


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="YOLO-Seg -> X-AnyLabeling JSON"
    )

    parser.add_argument(
        "--images",
        required=True,
        help="图片目录"
    )

    parser.add_argument(
        "--labels",
        required=True,
        help="YOLO-Seg labels 目录"
    )

    parser.add_argument(
        "--classes",
        required=True,
        help="classes.txt"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="输出 JSON 目录"
    )

    args = parser.parse_args()

    convert_dataset(
        images_dir=args.images,
        labels_dir=args.labels,
        output_dir=args.output,
        classes_path=args.classes,
    )


if __name__ == "__main__":
    main()

"""
python 3.yolo_seg_to_xanylabeling_json.py \
    --images /data/database/aws_origin_sample/sam3_person_train_dataset/filtered_dataset/images \
    --labels /data/database/aws_origin_sample/sam3_person_train_dataset/filtered_dataset/labels \
    --classes /data/database/aws_origin_sample/sam3_person_train_dataset/filtered_dataset/classes.txt \
    --output /data/database/aws_origin_sample/sam3_person_train_dataset/filtered_dataset/json

"""
