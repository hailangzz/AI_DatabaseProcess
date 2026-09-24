#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO-Segmentation 训练集样本筛选工具

功能：

1. 给定 images / labels 文件夹
2. 读取 YOLO Segmentation Label
3. 指定 MARK_CLASS_ID
4. 计算 mark 目标 Polygon 的外接矩形：
       x_min
       x_max
       y_min
       y_max

5. 根据原始图片尺寸：
       1280 x 720

   将 YOLO 归一化坐标转换成实际像素坐标。

6. 当存在一个 mark 目标满足：

       width  > MIN_WIDTH
       height > MIN_HEIGHT

   则将：

       image
       label

   一起复制到筛选目录。

7. 保证 images / labels 一一对应。

YOLO Segmentation Label 格式：

    class_id x1 y1 x2 y2 x3 y3 ...

坐标范围：

    0 ~ 1
"""

# ============================================================
# Imports
# ============================================================

import shutil
from pathlib import Path

# ============================================================
# 是否覆盖已经存在的文件
# ============================================================

OVERWRITE_EXISTING = False

# ============================================================
# 支持的图片格式
# ============================================================

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# 创建输出目录
# ============================================================


def create_output_dirs():
    """
    创建：

        OUTPUT_DIR/
        ├── images/
        └── labels/
    """

    output_images_dir = OUTPUT_DIR / "images"
    output_labels_dir = OUTPUT_DIR / "labels"

    output_images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_labels_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        output_images_dir,
        output_labels_dir,
    )


# ============================================================
# 获取图片路径
# ============================================================


def build_image_mapping():
    """
    建立：

        image_stem -> image_path

    例如：

        001.jpg
        001.png

    对应：

        "001" -> Path(...)
    """

    image_mapping = {}

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Images 目录不存在："
            f"{IMAGES_DIR}"
        )

    for image_path in IMAGES_DIR.iterdir():

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        stem = image_path.stem

        if stem in image_mapping:
            print(
                f"[WARNING] 存在同名不同扩展名图片："
                f"{stem}"
            )

            print(
                f"          已存在："
                f"{image_mapping[stem]}"
            )

            print(
                f"          当前："
                f"{image_path}"
            )

            continue

        image_mapping[stem] = image_path

    print(
        f"[INFO] Images 数量："
        f"{len(image_mapping)}"
    )

    return image_mapping


# ============================================================
# 解析单个 YOLO Seg Label
# ============================================================


def parse_yolo_segmentation_line(
        line,
        line_number,
        label_path,
):
    """
    解析：

        class_id x1 y1 x2 y2 ...

    返回：

        class_id
        points

    points：

        [
            (x1, y1),
            (x2, y2),
            ...
        ]

    坐标仍然保持为 0~1。
    """

    parts = line.strip().split()

    # --------------------------------------------------------
    # 至少：
    #
    # class_id
    # x1 y1
    # x2 y2
    # x3 y3
    #
    # = 7 个字段
    # --------------------------------------------------------

    if len(parts) < 7:
        print(
            f"[WARNING] Label 格式异常："
            f"{label_path}"
            f":{line_number}"
        )

        return None

    # --------------------------------------------------------
    # class_id
    # --------------------------------------------------------

    try:

        class_id = int(parts[0])

    except ValueError:

        print(
            f"[WARNING] class_id 无法解析："
            f"{label_path}"
            f":{line_number}"
        )

        return None

    coordinate_values = []

    # --------------------------------------------------------
    # 解析坐标
    # --------------------------------------------------------

    try:

        for value in parts[1:]:
            coordinate_values.append(
                float(value)
            )

    except ValueError:

        print(
            f"[WARNING] 坐标无法解析："
            f"{label_path}"
            f":{line_number}"
        )

        return None

    # --------------------------------------------------------
    # x / y 必须成对
    # --------------------------------------------------------

    if len(coordinate_values) % 2 != 0:
        print(
            f"[WARNING] Polygon 坐标数量不是偶数："
            f"{label_path}"
            f":{line_number}"
        )

        return None

    # --------------------------------------------------------
    # 至少 3 个点
    # --------------------------------------------------------

    if len(coordinate_values) < 6:
        print(
            f"[WARNING] Polygon 少于 3 个点："
            f"{label_path}"
            f":{line_number}"
        )

        return None

    # --------------------------------------------------------
    # 生成 point
    # --------------------------------------------------------

    points = []

    for i in range(
            0,
            len(coordinate_values),
            2,
    ):
        x = coordinate_values[i]
        y = coordinate_values[i + 1]

        points.append(
            (x, y)
        )

    return class_id, points


# ============================================================
# 计算 Polygon BBox
# ============================================================


def calculate_polygon_bbox(
        points,
):
    """
    根据归一化 Polygon 坐标计算：

        x_min
        y_min
        x_max
        y_max

    然后转换成实际像素。

    返回：

        x_min_px
        y_min_px
        x_max_px
        y_max_px
        width_px
        height_px
    """

    x_values = [
        point[0]
        for point in points
    ]

    y_values = [
        point[1]
        for point in points
    ]

    # --------------------------------------------------------
    # 归一化坐标 BBox
    # --------------------------------------------------------

    x_min = min(x_values)
    x_max = max(x_values)

    y_min = min(y_values)
    y_max = max(y_values)

    # --------------------------------------------------------
    # 转换为实际像素
    # --------------------------------------------------------

    x_min_px = x_min * IMAGE_WIDTH
    x_max_px = x_max * IMAGE_WIDTH

    y_min_px = y_min * IMAGE_HEIGHT
    y_max_px = y_max * IMAGE_HEIGHT

    # --------------------------------------------------------
    # 宽高
    # --------------------------------------------------------

    width_px = (
            x_max_px
            - x_min_px
    )

    height_px = (
            y_max_px
            - y_min_px
    )

    return (
        x_min_px,
        y_min_px,
        x_max_px,
        y_max_px,
        width_px,
        height_px,
    )


# ============================================================
# 判断 Label 是否满足筛选条件
# ============================================================


def check_label(
        label_path,
):
    """
    检查一个 Label 文件。

    只检查：

        MARK_CLASS_ID

    如果存在一个 mark：

        width  > MIN_WIDTH
        height > MIN_HEIGHT

    则返回 True。

    返回：

        matched
        max_width
        max_height
        mark_count
    """

    matched = False

    max_width = 0.0
    max_height = 0.0

    mark_count = 0

    try:

        with label_path.open(
                "r",
                encoding="utf-8",
        ) as f:

            for line_number, line in enumerate(
                    f,
                    start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                result = parse_yolo_segmentation_line(
                    line=line,
                    line_number=line_number,
                    label_path=label_path,
                )

                if result is None:
                    continue

                class_id, points = result

                # ------------------------------------------------
                # 只关注 MARK_CLASS_ID
                # ------------------------------------------------

                if class_id != MARK_CLASS_ID:
                    continue

                mark_count += 1

                (
                    x_min_px,
                    y_min_px,
                    x_max_px,
                    y_max_px,
                    width_px,
                    height_px,
                ) = calculate_polygon_bbox(
                    points
                )

                max_width = max(
                    max_width,
                    width_px,
                )

                max_height = max(
                    max_height,
                    height_px,
                )

                print(
                    f"[MARK] "
                    f"{label_path.name} | "
                    f"width={width_px:.1f}px | "
                    f"height={height_px:.1f}px"
                )

                # ------------------------------------------------
                # 满足条件
                # ------------------------------------------------

                if (
                        width_px > MIN_WIDTH
                        and height_px > MIN_HEIGHT
                ):
                    matched = True

                    print(
                        f"[MATCH] "
                        f"{label_path.name} | "
                        f"width={width_px:.1f}px > "
                        f"{MIN_WIDTH}px | "
                        f"height={height_px:.1f}px > "
                        f"{MIN_HEIGHT}px"
                    )

                    # ------------------------------------------------
                    # 已经满足条件，可以提前结束
                    # ------------------------------------------------

                    return (
                        True,
                        max_width,
                        max_height,
                        mark_count,
                    )

    except Exception as e:

        print(
            f"[ERROR] 读取 Label 失败："
            f"{label_path}"
        )

        print(
            f"[ERROR] {type(e).__name__}: {e}"
        )

        return (
            False,
            max_width,
            max_height,
            mark_count,
        )

    return (
        matched,
        max_width,
        max_height,
        mark_count,
    )


# ============================================================
# 复制文件
# ============================================================


def copy_file(
        source,
        destination,
):
    """
    文件复制。
    """

    if destination.exists():

        if not OVERWRITE_EXISTING:
            return True

    try:

        shutil.copy2(
            source,
            destination,
        )

        return True

    except Exception as e:

        print(
            f"[ERROR] 文件复制失败："
            f"{source}"
        )

        print(
            f"[ERROR] {type(e).__name__}: {e}"
        )

        return False


# ============================================================
# 主函数
# ============================================================


def main():
    print("\n")

    print("=" * 80)

    print(
        "YOLO-Segmentation "
        "训练集样本筛选"
    )

    print("=" * 80)

    print(
        f"[CONFIG] Images："
        f"{IMAGES_DIR}"
    )

    print(
        f"[CONFIG] Labels："
        f"{LABELS_DIR}"
    )

    print(
        f"[CONFIG] Output："
        f"{OUTPUT_DIR}"
    )

    print(
        f"[CONFIG] Image Size："
        f"{IMAGE_WIDTH} x {IMAGE_HEIGHT}"
    )

    print(
        f"[CONFIG] MARK_CLASS_ID："
        f"{MARK_CLASS_ID}"
    )

    print(
        f"[CONFIG] MIN_WIDTH："
        f"{MIN_WIDTH}px"
    )

    print(
        f"[CONFIG] MIN_HEIGHT："
        f"{MIN_HEIGHT}px"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # 检查 Labels
    # --------------------------------------------------------

    if not LABELS_DIR.exists():
        raise FileNotFoundError(
            f"Labels 目录不存在："
            f"{LABELS_DIR}"
        )

    # --------------------------------------------------------
    # 创建输出目录
    # --------------------------------------------------------

    (
        output_images_dir,
        output_labels_dir,
    ) = create_output_dirs()

    # --------------------------------------------------------
    # 建立 Image Mapping
    # --------------------------------------------------------

    image_mapping = build_image_mapping()

    # --------------------------------------------------------
    # 获取所有 Label
    # --------------------------------------------------------

    label_paths = sorted(
        LABELS_DIR.glob("*.txt")
    )

    total_labels = len(label_paths)

    print(
        f"[INFO] Labels 数量："
        f"{total_labels}"
    )

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    matched_count = 0
    no_match_count = 0
    missing_image_count = 0
    failed_count = 0

    # --------------------------------------------------------
    # 逐个 Label 处理
    # --------------------------------------------------------

    for index, label_path in enumerate(
            label_paths,
            start=1,
    ):

        print("\n")

        print("-" * 80)

        print(
            f"[{index}/{total_labels}] "
            f"{label_path.name}"
        )

        # ----------------------------------------------------
        # 查找对应图片
        # ----------------------------------------------------

        image_path = image_mapping.get(
            label_path.stem
        )

        if image_path is None:
            print(
                f"[WARNING] 找不到对应图片："
                f"{label_path}"
            )

            missing_image_count += 1

            continue

        # ----------------------------------------------------
        # 检查 Label
        # ----------------------------------------------------

        (
            matched,
            max_width,
            max_height,
            mark_count,
        ) = check_label(
            label_path
        )

        # ----------------------------------------------------
        # 没有 Mark
        # ----------------------------------------------------

        if mark_count == 0:
            print(
                f"[SKIP] "
                f"没有 class={MARK_CLASS_ID} "
                f"目标"
            )

            no_match_count += 1

            continue

        # ----------------------------------------------------
        # 不满足尺寸
        # ----------------------------------------------------

        if not matched:
            print(
                f"[SKIP] "
                f"不满足尺寸条件 | "
                f"max_width={max_width:.1f}px | "
                f"max_height={max_height:.1f}px"
            )

            no_match_count += 1

            continue

        # ----------------------------------------------------
        # 满足条件
        # ----------------------------------------------------

        output_image_path = (
                output_images_dir
                / image_path.name
        )

        output_label_path = (
                output_labels_dir
                / label_path.name
        )

        # ----------------------------------------------------
        # 复制图片
        # ----------------------------------------------------

        image_success = copy_file(
            source=image_path,
            destination=output_image_path,
        )

        # ----------------------------------------------------
        # 复制 Label
        # ----------------------------------------------------

        label_success = copy_file(
            source=label_path,
            destination=output_label_path,
        )

        # ----------------------------------------------------
        # 必须 image + label 同时成功
        # ----------------------------------------------------

        if image_success and label_success:

            matched_count += 1

            print(
                f"[SUCCESS] 已复制："
                f"{image_path.name}"
            )

        else:

            failed_count += 1

            # ------------------------------------------------
            # 防止产生 image 无 label
            # ------------------------------------------------

            if (
                    output_image_path.exists()
                    and not output_label_path.exists()
            ):

                try:

                    output_image_path.unlink()

                except Exception:
                    pass

            # ------------------------------------------------
            # 防止产生 label 无 image
            # ------------------------------------------------

            if (
                    output_label_path.exists()
                    and not output_image_path.exists()
            ):

                try:

                    output_label_path.unlink()

                except Exception:
                    pass

    # ========================================================
    # 最终统计
    # ========================================================

    print("\n")

    print("=" * 80)

    print(
        "YOLO-Segmentation "
        "样本筛选完成"
    )

    print("=" * 80)

    print(
        f"[RESULT] Label 总数："
        f"{total_labels}"
    )

    print(
        f"[RESULT] 满足条件："
        f"{matched_count}"
    )

    print(
        f"[RESULT] 不满足条件："
        f"{no_match_count}"
    )

    print(
        f"[RESULT] 找不到图片："
        f"{missing_image_count}"
    )

    print(
        f"[RESULT] 处理失败："
        f"{failed_count}"
    )

    print("-" * 80)

    print(
        f"[RESULT] Images："
        f"{output_images_dir}"
    )

    print(
        f"[RESULT] Labels："
        f"{output_labels_dir}"
    )

    print("=" * 80)


# ============================================================
# 配置
# ============================================================

# ------------------------------------------------------------
# 原始 Images
# ------------------------------------------------------------

IMAGES_DIR = Path(
    "/data/database/aws_origin_sample/sam3_person_train_dataset/images"
)

# ------------------------------------------------------------
# 原始 Labels
# ------------------------------------------------------------

LABELS_DIR = Path(
    "/data/database/aws_origin_sample/sam3_person_train_dataset/labels"
)

# ------------------------------------------------------------
# 筛选后的 Dataset
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    "/data/database/aws_origin_sample/sam3_person_train_dataset/filtered_dataset"
)

# ============================================================
# 原始图像尺寸
# ============================================================

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# ============================================================
# Mark 类别
# ============================================================

# 例如：
#
# 0 = carpet
# 1 = wire
# 2 = liquid
# 3 = plasticbag
#
# 根据你的实际 classes.txt 修改。
# ============================================================

MARK_CLASS_ID = 0

# ============================================================
# 筛选阈值
# ============================================================

# mark 目标实际像素宽度必须：
#
#     > MIN_WIDTH
#
# ============================================================

MIN_WIDTH = 64

# mark 目标实际像素高度必须：
#
#     > MIN_HEIGHT
#
# ============================================================

MIN_HEIGHT = 128

# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()
