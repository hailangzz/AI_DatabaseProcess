import json
import os

import cv2
import numpy as np


# ================== 基础工具 ==================

def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def normalize_point(x, y, w, h):
    """像素坐标归一化"""
    return x / w, y / h


def polygon_to_bbox(points):
    """根据多边形计算 YOLO bbox，使用原始轮廓"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    bw = xmax - xmin
    bh = ymax - ymin

    return cx, cy, bw, bh


# ================== 轮廓简化模块 ==================

def simplify_polygon(points, epsilon=1.0):
    """
    使用 OpenCV approxPolyDP 简化多边形轮廓。

    参数：
        points:
            原始轮廓点，格式为 [[x1, y1], [x2, y2], ...]
        epsilon:
            轮廓简化容差，单位为原图像素。
            数值越大，简化程度越高。

    返回：
        简化后的轮廓点列表。
        如果输入无效或简化结果不合法，则返回原始轮廓。
    """

    # 保留原始轮廓，避免修改输入数据
    original_points = [
        [float(p[0]), float(p[1])]
        for p in points
    ]

    # 多边形至少需要 3 个点
    if len(original_points) < 3:
        return original_points

    # 检查坐标是否合法
    if not np.isfinite(np.asarray(original_points)).all():
        return original_points

    # epsilon 为 0 时不进行简化
    if epsilon <= 0:
        return original_points

    contour = np.asarray(
        original_points,
        dtype=np.float32
    ).reshape(-1, 1, 2)

    try:
        simplified = cv2.approxPolyDP(
            contour,
            epsilon=float(epsilon),
            closed=True
        )

        simplified_points = simplified.reshape(-1, 2)

        # 简化后至少需要 3 个点
        if len(simplified_points) < 3:
            return original_points

        # 检查简化结果是否包含非法数值
        if not np.isfinite(simplified_points).all():
            return original_points

        # 检查简化后的多边形面积，避免退化轮廓
        original_area = abs(
            cv2.contourArea(contour)
        )
        simplified_area = abs(
            cv2.contourArea(simplified)
        )

        if simplified_area <= 1e-6:
            return original_points

        # 原始轮廓本身具有有效面积时，避免面积异常变化
        if original_area > 1e-6:
            area_ratio = simplified_area / original_area

            # 如果面积变化超过 50%，保留原始轮廓
            if area_ratio < 0.5 or area_ratio > 1.5:
                return original_points

        return simplified_points.tolist()

    except (cv2.error, ValueError, TypeError):
        # 出现异常时回退到原始轮廓
        return original_points


# ================== JSON 转 YOLO ==================

def process_json(
        json_path,
        seg_out_path,
        bbox_out_path,
        epsilon=1.0
):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]

    if img_w <= 0 or img_h <= 0:
        raise ValueError(
            f"图片尺寸无效：{json_path}, "
            f"width={img_w}, height={img_h}"
        )

    seg_lines = []
    bbox_lines = []

    # 统计当前图片的简化效果
    original_total_points = 0
    simplified_total_points = 0
    simplified_objects = 0
    skipped_objects = 0

    for shape in data.get("shapes", []):
        label = shape.get("label")

        if label not in CLASS_MAP:
            continue

        class_id = CLASS_MAP[label]
        points = shape.get("points", [])

        # 忽略无效的多边形
        if not isinstance(points, list) or len(points) < 3:
            skipped_objects += 1
            continue

        # 统一为浮点坐标
        try:
            original_points = [
                [float(p[0]), float(p[1])]
                for p in points
            ]
        except (TypeError, ValueError, IndexError):
            skipped_objects += 1
            continue

        if not np.isfinite(
                np.asarray(original_points, dtype=np.float64)
        ).all():
            skipped_objects += 1
            continue

        # ===== 1. YOLO-seg：简化轮廓 =====

        simplified_points = simplify_polygon(
            original_points,
            epsilon=epsilon
        )

        original_count = len(original_points)
        simplified_count = len(simplified_points)

        original_total_points += original_count
        simplified_total_points += simplified_count

        if simplified_count < original_count:
            simplified_objects += 1

        seg_line = [str(class_id)]

        for x, y in simplified_points:
            nx, ny = normalize_point(
                x, y, img_w, img_h
            )

            seg_line.append(f"{nx:.6f}")
            seg_line.append(f"{ny:.6f}")

        seg_lines.append(" ".join(seg_line))

        # ===== 2. YOLO-bbox：仍使用原始轮廓 =====

        cx, cy, bw, bh = polygon_to_bbox(
            original_points
        )

        cx, cy = normalize_point(
            cx, cy, img_w, img_h
        )

        bw /= img_w
        bh /= img_h

        bbox_line = (
            f"{class_id} "
            f"{cx:.6f} "
            f"{cy:.6f} "
            f"{bw:.6f} "
            f"{bh:.6f}"
        )

        bbox_lines.append(bbox_line)

    # ===== 写入文件 =====

    with open(seg_out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(seg_lines))

    with open(bbox_out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(bbox_lines))

    return {
        "objects": len(seg_lines),
        "simplified_objects": simplified_objects,
        "skipped_objects": skipped_objects,
        "original_points": original_total_points,
        "simplified_points": simplified_total_points,
    }


# ================== 主流程 ==================

def main():
    yolov8_label_dir = os.path.join(
        OUTPUT_DIR,
        "yolov8_labels"
    )

    seg_dir = os.path.join(
        yolov8_label_dir,
        "seg"
    )

    bbox_dir = os.path.join(
        yolov8_label_dir,
        "bbox"
    )

    ensure_dir(seg_dir)
    ensure_dir(bbox_dir)

    json_files = sorted(
        file for file in os.listdir(INPUT_DIR)
        if file.lower().endswith(".json")
    )

    if not json_files:
        print(f"⚠️ 未找到 JSON 文件：{INPUT_DIR}")
        return

    # 全局统计
    total_objects = 0
    total_simplified_objects = 0
    total_skipped_objects = 0
    total_original_points = 0
    total_simplified_points = 0
    failed_files = 0

    print(f"输入目录：{INPUT_DIR}")
    print(f"输出目录：{yolov8_label_dir}")
    print(f"简化容差：{SIMPLIFY_EPSILON} 像素")
    print(f"待处理 JSON 数量：{len(json_files)}")
    print("-" * 60)

    for index, file in enumerate(json_files, start=1):
        json_path = os.path.join(
            INPUT_DIR,
            file
        )

        name = os.path.splitext(file)[0]

        seg_out = os.path.join(
            seg_dir,
            name + ".txt"
        )

        bbox_out = os.path.join(
            bbox_dir,
            name + ".txt"
        )

        try:
            stats = process_json(
                json_path=json_path,
                seg_out_path=seg_out,
                bbox_out_path=bbox_out,
                epsilon=SIMPLIFY_EPSILON
            )

            total_objects += stats["objects"]
            total_simplified_objects += (
                stats["simplified_objects"]
            )
            total_skipped_objects += (
                stats["skipped_objects"]
            )
            total_original_points += (
                stats["original_points"]
            )
            total_simplified_points += (
                stats["simplified_points"]
            )

        except Exception as e:
            failed_files += 1
            print(f"❌ 处理失败：{file}，原因：{e}")
            continue

        if index % 100 == 0 or index == len(json_files):
            print(
                f"进度：{index}/{len(json_files)}"
            )

    # ===== 输出统计 =====

    if total_original_points > 0:
        reduction_ratio = (
                                  1 - total_simplified_points
                                  / total_original_points
                          ) * 100

        average_original = (
                total_original_points / max(total_objects, 1)
        )

        average_simplified = (
                total_simplified_points / max(total_objects, 1)
        )
    else:
        reduction_ratio = 0.0
        average_original = 0.0
        average_simplified = 0.0

    print("\n" + "=" * 60)
    print("✅ 转换完成！")
    print(f"处理文件数：{len(json_files)}")
    print(f"失败文件数：{failed_files}")
    print(f"有效目标数：{total_objects}")
    print(f"跳过的无效目标数：{total_skipped_objects}")
    print(f"点数减少的目标数：{total_simplified_objects}")
    print(f"原始轮廓总点数：{total_original_points}")
    print(f"简化后总点数：{total_simplified_points}")
    print(f"平均每个目标原始点数：{average_original:.2f}")
    print(f"平均每个目标简化后点数：{average_simplified:.2f}")
    print(f"轮廓点数减少比例：{reduction_ratio:.2f}%")
    print("=" * 60)


# ====== 配置 ======

# OUTPUT_DIR = "/data/database/AITotal_Real_Customer_Database/Real_Wire_Customer_Database/date0519/WireSampleFolder"  # 输出目录
# INPUT_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_DIR = "/data/database/aws_origin_sample/sam3_person_train_dataset/filtered_dataset"  # 输出目录
INPUT_DIR = os.path.join(OUTPUT_DIR, "images")

# 类别映射（根据你的实际类别修改）
CLASS_MAP = {
    # "liquid": 0,
    # "hand":0,
    # "carpet":0,
    # "wire":0
    # "plastic": 0
    "person": 0
}

# 轮廓简化容差，单位为原图像素
# 设为 0 可关闭简化
SIMPLIFY_EPSILON = 1.0

if __name__ == "__main__":
    main()
