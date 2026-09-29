#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO-Segmentation 训练集样本筛选工具

功能：

1. 给定 images / labels 文件夹
2. 读取 YOLO Segmentation Label
3. 指定 MARK_CLASS_ID
4. 计算 mark 目标 Polygon 的外接矩形
5. 根据原始图片尺寸 1280 x 720
   将 YOLO 归一化坐标转换成实际像素坐标
6. 当存在一个 mark 目标满足：
       width  > MIN_WIDTH
       height > MIN_HEIGHT
   则将 image + NMS 后的 label
   一起复制到筛选目录
7. 对单张样本中的 MARK_CLASS_ID 标注执行 Polygon NMS：
   - Polygon IoU 过大 -> 删除面积较小的目标
   - 大 mark 对小 mark 覆盖率过高 -> 删除面积较小的目标
8. 保证 images / labels 一一对应
9. 使用 tqdm 显示处理进度
10. 不修改原始 Label
"""

# ============================================================
# Imports
# ============================================================

import shutil
from pathlib import Path

from shapely.geometry import Polygon
from tqdm import tqdm

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
# NMS 参数
# ============================================================

# Polygon IoU 阈值
#
# 两个 mark：
#
#     intersection_area / union_area
#
# 大于等于该阈值时：
#
#     删除面积较小的 mark
#
NMS_IOU_THRESHOLD = 0.70

# ============================================================
# 小目标覆盖率阈值
# ============================================================

# 大 mark 对小 mark 的覆盖率：
#
#     intersection_area / small_mark_area
#
# 大于等于该阈值时：
#
#     删除小 mark
#
SMALL_MARK_COVERAGE_THRESHOLD = 0.80


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
    """

    image_mapping = {}

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Images 目录不存在：{IMAGES_DIR}"
        )

    for image_path in IMAGES_DIR.iterdir():

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        stem = image_path.stem

        # 同名不同扩展名时保留第一个
        if stem in image_mapping:
            continue

        image_mapping[stem] = image_path

    return image_mapping


# ============================================================
# 解析单个 YOLO Seg Label
# ============================================================

def parse_yolo_segmentation_line(
        line,
):
    """
    解析：

        class_id x1 y1 x2 y2 x3 y3 ...

    返回：

        class_id
        points

    points：

        [
            (x1, y1),
            (x2, y2),
            ...
        ]

    坐标保持为 0~1。
    """

    parts = line.strip().split()

    if len(parts) < 7:
        return None

    try:
        class_id = int(parts[0])
    except ValueError:
        return None

    coordinate_values = []

    try:

        for value in parts[1:]:
            coordinate_values.append(
                float(value)
            )

    except ValueError:

        return None

    if len(coordinate_values) % 2 != 0:
        return None

    if len(coordinate_values) < 6:
        return None

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

    并转换成实际像素。
    """

    x_values = [
        point[0]
        for point in points
    ]

    y_values = [
        point[1]
        for point in points
    ]

    x_min = min(x_values)
    x_max = max(x_values)

    y_min = min(y_values)
    y_max = max(y_values)

    x_min_px = x_min * IMAGE_WIDTH
    x_max_px = x_max * IMAGE_WIDTH

    y_min_px = y_min * IMAGE_HEIGHT
    y_max_px = y_max * IMAGE_HEIGHT

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
# 创建 Shapely Polygon
# ============================================================

def create_polygon(
        points,
):
    """
    将 YOLO 归一化 Polygon
    转换成 Shapely Polygon。
    """

    try:

        polygon = Polygon(points)

        # 修复可能存在的自交 Polygon
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if polygon.is_empty:
            return None

        if polygon.area <= 0:
            return None

        return polygon

    except Exception:
        return None


# ============================================================
# 计算 Polygon 面积
# ============================================================

def calculate_polygon_area(
        points,
):
    """
    计算 Polygon 面积。

    使用 Shapely。
    """

    polygon = create_polygon(points)

    if polygon is None:
        return 0.0

    return polygon.area


# ============================================================
# Polygon NMS
# ============================================================

def polygon_nms(
        mark_records,
):
    """
    对同一个样本中的 MARK_CLASS_ID
    执行 Polygon NMS。

    规则：

    1. 面积大的目标优先保留。

    2. 如果两个目标：

       IoU >= NMS_IOU_THRESHOLD

       删除面积较小的目标。

    3. 或者：

       intersection_area / small_mark_area
       >= SMALL_MARK_COVERAGE_THRESHOLD

       删除面积较小的目标。

    参数：

        mark_records：

            [
                {
                    "line_index": int,
                    "line": str,
                    "points": list,
                    "polygon": Polygon,
                    "area": float,
                }
            ]

    返回：

        保留下来的 records
    """

    if len(mark_records) <= 1:
        return mark_records

    # --------------------------------------------------------
    # 按面积从大到小排序
    # --------------------------------------------------------

    sorted_records = sorted(
        mark_records,
        key=lambda item: item["area"],
        reverse=True,
    )

    keep_records = []

    for current in sorted_records:

        current_polygon = current["polygon"]
        current_area = current["area"]

        should_remove = False

        # ----------------------------------------------------
        # 当前目标只与已经保留的“大目标”比较
        # ----------------------------------------------------

        for kept in keep_records:

            kept_polygon = kept["polygon"]
            kept_area = kept["area"]

            try:

                intersection_area = (
                    current_polygon
                    .intersection(kept_polygon)
                    .area
                )

            except Exception:

                continue

            if intersection_area <= 0:
                continue

            # ------------------------------------------------
            # IoU
            # ------------------------------------------------

            union_area = (
                    current_area
                    + kept_area
                    - intersection_area
            )

            if union_area <= 0:
                continue

            iou = (
                    intersection_area
                    / union_area
            )

            # ------------------------------------------------
            # 大 mark 对小 mark 的覆盖率
            # ------------------------------------------------

            small_mark_coverage = (
                    intersection_area
                    / current_area
            )

            # ------------------------------------------------
            # NMS 判断
            # ------------------------------------------------

            if (
                    iou >= NMS_IOU_THRESHOLD
                    or
                    small_mark_coverage
                    >= SMALL_MARK_COVERAGE_THRESHOLD
            ):
                should_remove = True
                break

        # ----------------------------------------------------
        # 保留当前目标
        # ----------------------------------------------------

        if not should_remove:
            keep_records.append(
                current
            )

    return keep_records


# ============================================================
# 对 Label 执行 NMS
# ============================================================

def nms_label(
        label_path,
):
    """
    对一个 Label 文件执行 NMS。

    只对：

        MARK_CLASS_ID

    执行 NMS。

    其他 class 的 Label 原样保留。

    返回：

        filtered_lines
    """

    try:

        with label_path.open(
                "r",
                encoding="utf-8",
        ) as f:

            lines = f.readlines()

    except Exception:

        return None

    mark_records = []

    # --------------------------------------------------------
    # 记录所有 Label
    # --------------------------------------------------------

    parsed_records = []

    for line_index, line in enumerate(lines):

        stripped_line = line.strip()

        if not stripped_line:
            continue

        result = parse_yolo_segmentation_line(
            stripped_line
        )

        if result is None:
            # 无法解析的行原样保留
            parsed_records.append({
                "line_index": line_index,
                "line": line,
                "is_mark": False,
            })

            continue

        class_id, points = result

        # ----------------------------------------------------
        # 非 MARK_CLASS_ID
        # ----------------------------------------------------

        if class_id != MARK_CLASS_ID:
            parsed_records.append({
                "line_index": line_index,
                "line": line,
                "is_mark": False,
            })

            continue

        # ----------------------------------------------------
        # MARK_CLASS_ID
        # ----------------------------------------------------

        polygon = create_polygon(
            points
        )

        if polygon is None:
            # Polygon 无效时原样保留
            parsed_records.append({
                "line_index": line_index,
                "line": line,
                "is_mark": False,
            })

            continue

        area = polygon.area

        record = {
            "line_index": line_index,
            "line": line,
            "points": points,
            "polygon": polygon,
            "area": area,
        }

        mark_records.append(
            record
        )

    # --------------------------------------------------------
    # 执行 NMS
    # --------------------------------------------------------

    keep_mark_records = polygon_nms(
        mark_records
    )

    keep_line_indices = {
        record["line_index"]
        for record in keep_mark_records
    }

    # --------------------------------------------------------
    # 重新生成 Label
    # --------------------------------------------------------

    filtered_lines = []

    for line_index, line in enumerate(lines):

        stripped_line = line.strip()

        if not stripped_line:
            continue

        result = parse_yolo_segmentation_line(
            stripped_line
        )

        # 无法解析
        if result is None:
            filtered_lines.append(
                line
            )

            continue

        class_id, _ = result

        # ----------------------------------------------------
        # 非 MARK_CLASS_ID
        # ----------------------------------------------------

        if class_id != MARK_CLASS_ID:
            filtered_lines.append(
                line
            )

            continue

        # ----------------------------------------------------
        # MARK_CLASS_ID
        #
        # 只保留 NMS 后的目标
        # ----------------------------------------------------

        if line_index in keep_line_indices:
            filtered_lines.append(
                line
            )

    return filtered_lines


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

            for line in f:

                line = line.strip()

                if not line:
                    continue

                result = parse_yolo_segmentation_line(
                    line
                )

                if result is None:
                    continue

                class_id, points = result

                if class_id != MARK_CLASS_ID:
                    continue

                mark_count += 1

                (
                    _,
                    _,
                    _,
                    _,
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

                # ------------------------------------------------
                # 原来的筛选条件完全保持不变
                # ------------------------------------------------

                if (
                        width_px > MIN_WIDTH
                        and
                        height_px > MIN_HEIGHT
                ):
                    matched = True

                    return (
                        True,
                        max_width,
                        max_height,
                        mark_count,
                    )

    except Exception:

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

    except Exception:

        return False


# ============================================================
# 主函数
# ============================================================

def main():
    # --------------------------------------------------------
    # 检查 Labels
    # --------------------------------------------------------

    if not LABELS_DIR.exists():
        raise FileNotFoundError(
            f"Labels 目录不存在：{LABELS_DIR}"
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

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    matched_count = 0
    no_match_count = 0
    missing_image_count = 0
    failed_count = 0

    nms_removed_count = 0

    # --------------------------------------------------------
    # tqdm
    # --------------------------------------------------------

    progress = tqdm(
        label_paths,
        total=total_labels,
        desc="Processing",
        unit="label",
        dynamic_ncols=True,
    )

    # --------------------------------------------------------
    # 逐个 Label
    # --------------------------------------------------------

    for label_path in progress:

        # ----------------------------------------------------
        # 查找对应图片
        # ----------------------------------------------------

        image_path = image_mapping.get(
            label_path.stem
        )

        if image_path is None:
            missing_image_count += 1

            progress.set_postfix(
                matched=matched_count,
                nms_removed=nms_removed_count,
                missing=missing_image_count,
                failed=failed_count,
            )

            continue

        # ----------------------------------------------------
        # 检查 Label
        # ----------------------------------------------------

        (
            matched,
            _,
            _,
            mark_count,
        ) = check_label(
            label_path
        )

        # ----------------------------------------------------
        # 没有 Mark
        # ----------------------------------------------------

        if mark_count == 0:
            no_match_count += 1

            progress.set_postfix(
                matched=matched_count,
                nms_removed=nms_removed_count,
                missing=missing_image_count,
                failed=failed_count,
            )

            continue

        # ----------------------------------------------------
        # 不满足尺寸
        # ----------------------------------------------------

        if not matched:
            no_match_count += 1

            progress.set_postfix(
                matched=matched_count,
                nms_removed=nms_removed_count,
                missing=missing_image_count,
                failed=failed_count,
            )

            continue

        # ====================================================
        # 满足筛选条件
        # 开始生成 NMS 后的 Label
        # ====================================================

        filtered_lines = nms_label(
            label_path
        )

        if filtered_lines is None:
            failed_count += 1

            progress.set_postfix(
                matched=matched_count,
                nms_removed=nms_removed_count,
                missing=missing_image_count,
                failed=failed_count,
            )

            continue

        # ----------------------------------------------------
        # 计算 NMS 删除数量
        # ----------------------------------------------------

        try:

            with label_path.open(
                    "r",
                    encoding="utf-8",
            ) as f:

                original_line_count = sum(
                    1
                    for line in f
                    if line.strip()
                )

            filtered_line_count = len(
                filtered_lines
            )

            nms_removed_count += (
                    original_line_count
                    - filtered_line_count
            )

        except Exception:
            pass

        # ----------------------------------------------------
        # 输出路径
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
        # 复制 Image
        # ----------------------------------------------------

        image_success = copy_file(
            source=image_path,
            destination=output_image_path,
        )

        if not image_success:
            failed_count += 1

            progress.set_postfix(
                matched=matched_count,
                nms_removed=nms_removed_count,
                missing=missing_image_count,
                failed=failed_count,
            )

            continue

        # ----------------------------------------------------
        # 保存 NMS 后 Label
        # ----------------------------------------------------

        try:

            if (
                    output_label_path.exists()
                    and not OVERWRITE_EXISTING
            ):

                label_success = True

            else:

                with output_label_path.open(
                        "w",
                        encoding="utf-8",
                ) as f:

                    f.writelines(
                        filtered_lines
                    )

                label_success = True

        except Exception:

            label_success = False

        # ----------------------------------------------------
        # Image + Label 必须同时成功
        # ----------------------------------------------------

        if image_success and label_success:

            matched_count += 1

        else:

            failed_count += 1

            # ------------------------------------------------
            # 防止 image 无 label
            # ------------------------------------------------

            if (
                    output_image_path.exists()
                    and
                    not output_label_path.exists()
            ):

                try:
                    output_image_path.unlink()
                except Exception:
                    pass

            # ------------------------------------------------
            # 防止 label 无 image
            # ------------------------------------------------

            if (
                    output_label_path.exists()
                    and
                    not output_image_path.exists()
            ):

                try:
                    output_label_path.unlink()
                except Exception:
                    pass

        # ----------------------------------------------------
        # 更新进度条
        # ----------------------------------------------------

        progress.set_postfix(
            matched=matched_count,
            nms_removed=nms_removed_count,
            missing=missing_image_count,
            failed=failed_count,
        )

    # ========================================================
    # 最终结果
    # ========================================================

    progress.close()

    tqdm.write("")
    tqdm.write("处理完成")
    tqdm.write(
        f"Label 总数：{total_labels}"
    )
    tqdm.write(
        f"满足条件：{matched_count}"
    )
    tqdm.write(
        f"不满足条件：{no_match_count}"
    )
    tqdm.write(
        f"找不到图片：{missing_image_count}"
    )
    tqdm.write(
        f"处理失败：{failed_count}"
    )
    tqdm.write(
        f"NMS 删除标注：{nms_removed_count}"
    )
    tqdm.write(
        f"Images：{output_images_dir}"
    )
    tqdm.write(
        f"Labels：{output_labels_dir}"
    )


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
# 0 = person
# 1 = carpet
# 2 = wire
# 3 = liquid
# 4 = plasticbag
#
# 根据你的实际 classes.txt 修改。

MARK_CLASS_ID = 0

# ============================================================
# 筛选阈值
# ============================================================

MIN_WIDTH = 64
MIN_HEIGHT = 128

# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()
