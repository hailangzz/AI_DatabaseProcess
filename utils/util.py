import os, glob
import cv2
import numpy as np
import shutil
from collections import defaultdict
import json


def read_name_list(save_image_direct_path):
    files_name_list = [os.path.basename(f) for f in glob.glob(save_image_direct_path + "/*")]
    return files_name_list


def mark_to_detect(mask_dir):  # 根据mask的信息，创建mask目标对应的最小外接矩形（目标检测框）
    # min_area 太小的话，未来标注的标签会特别多，误差很大
    min_area = 100  # 小于该像素面积的目标将被忽略 (测试结果显示，最小像素为100时，检测框标注文件，标注效果很好)
    save_dir = os.path.join(mask_dir[:mask_dir.rfind("/")], "labels")
    os.makedirs(save_dir, exist_ok=True)

    for mask_name in os.listdir(mask_dir):
        if not mask_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue

        mask_path = os.path.join(mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        h, w = mask.shape

        label_path = os.path.join(save_dir, os.path.splitext(mask_name)[0] + ".txt")
        with open(label_path, "w") as f:
            # 找出所有非零类别（跳过背景0）
            for cls_id in np.unique(mask):
                if cls_id == 0:
                    continue

                # 生成该类别的二值掩码
                binary = (mask == cls_id).astype(np.uint8)

                # 查找所有连通区域
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < min_area:
                        continue  # 跳过小目标

                    x, y, bw, bh = cv2.boundingRect(cnt)

                    # 转换为 YOLO 格式（归一化）
                    x_center = (x + bw / 2) / w
                    y_center = (y + bh / 2) / h
                    norm_w = bw / w
                    norm_h = bh / h

                    f.write(f"{int(0)} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    print(f"✅ 转换完成！YOLO标签已保存到：{save_dir}")
    print(f"（已过滤掉面积小于 {min_area} 像素的目标）")


def use_yolo_label_plot_box(image_path):
    image_dir = image_path  # 原始图像文件夹
    label_dir = os.path.join(image_dir[:image_dir.rfind("/")], "labels")  # YOLO标签文件夹
    output_dir = os.path.join(image_dir[:image_dir.rfind("/")], "image_plot_box")  # 输出文件夹
    class_names = ["ElectricWires"]  # 可选: 类别名列表，如 ["person", "car", "dog"]
    os.makedirs(output_dir, exist_ok=True)

    # 颜色生成函数
    def get_color(idx):
        import random
        random.seed(idx)
        return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

    # 遍历所有图片
    for img_name in os.listdir(image_dir):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 无法读取图像：{img_path}")
            continue

        h, w, _ = img.shape

        # 如果没有对应标注文件，跳过
        if not os.path.exists(label_path):
            print(f"⚠️ 未找到标签文件：{label_path}")
            continue

        # 读取标签
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                cls_id, x_center, y_center, bw, bh = map(float, parts)
                cls_id = int(cls_id)

                # 转为像素坐标
                x_center *= w
                y_center *= h
                bw *= w
                bh *= h

                xmin = int(x_center - bw / 2)
                ymin = int(y_center - bh / 2)
                xmax = int(x_center + bw / 2)
                ymax = int(y_center + bh / 2)

                # 颜色与标签名
                color = get_color(cls_id)
                label_text = str(cls_id) if class_names is None else class_names[cls_id]

                # 绘制框
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(img, label_text, (xmin, max(ymin - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 保存可视化结果
        save_path = os.path.join(output_dir, img_name)
        cv2.imwrite(save_path, img)
        print(f"✅ 已保存标注图像：{save_path}")

    print("🎯 全部图像可视化完成！")
    pass


def draw_yolo_boxes(img, boxes, save_path="mosaic_pro.jpg", color=(0, 255, 0), thickness=2):
    """
    在图像上根据 YOLO 格式目标框绘制矩形框

    参数：
        img: numpy.ndarray, 原始图像矩阵 (H, W, C)
        boxes: list[np.ndarray] 或 list[list[float]]
        save_path: 图像保存路径
               YOLO 格式的目标框数组，每个元素为 [cls_id, x_center, y_center, width, height]
        color: tuple(int), 框的颜色 (B, G, R)
        thickness: int, 框线条粗细
    返回：
        绘制了框的图像
    """
    h, w = img.shape[:2]
    img_copy = img.copy()

    for box in boxes:
        cls_id, x_center, y_center, bw, bh = box

        # 转换为像素坐标
        x_center *= w
        y_center *= h
        bw *= w
        bh *= h

        # 计算左上角和右下角坐标
        x1 = int(x_center - bw / 2)
        y1 = int(y_center - bh / 2)
        x2 = int(x_center + bw / 2)
        y2 = int(y_center + bh / 2)
        # 绘制矩形框
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        # 绘制类别文本
        cv2.putText(img_copy, f"ID:{int(cls_id)}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 4)

    cv2.imwrite(save_path, img_copy)

    return img_copy


def draw_single_image_yolo_boxes(image_path, label_path, class_names=None):
    """
    image_path: 图片路径
    label_path: YOLO txt 标注路径
    class_names: 类别名称列表，例如 ["line", "cable"]；可以不用
    """

    # 1. 加载图片
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w = img.shape[:2]

    # 2. 读取 YOLO 标签
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls_id = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:5])

        # YOLO 格式 → 像素坐标
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        # 3. 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 写类别
        if class_names:
            text = class_names[cls_id]
        else:
            text = str(cls_id)

        cv2.putText(img, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 4. 显示图片
    return img


def resize_long_edge_image(image, target_long=1280):
    """
    按长边缩放图像，并同步调整YOLO标注

    Args:
        image (np.ndarray): 输入图像 (H, W, 3)
        target_long (int): 缩放后的长边尺寸 640、960、1024、1280 （32的倍数）

    Returns:
        resized_img: 缩放后的图像
    """
    h, w = image.shape[:2]
    scale = target_long / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    # 1️⃣ 图像缩放
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return resized


def move_batch_image_to_direct(source_dir=r"", target_dir=r""):
    os.makedirs(target_dir, exist_ok=True)
    # 定义允许的图片后缀
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

    # 遍历源目录
    for filename in os.listdir(source_dir):
        if filename.lower().endswith(image_extensions):
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename)

            # 移动文件
            shutil.move(source_path, target_path)
            print(f"已移动: {filename}")

    print("所有图片已移动完成！")


def iou(box1, box2):
    """计算两个框的IoU，输入为[x_min, y_min, x_max, y_max]"""
    x1, y1 = np.maximum(box1[:2], box2[:2])
    x2, y2 = np.minimum(box1[2:], box2[2:])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0


def merge_boxes(box1, box2):
    """合并两个框为一个更大的框"""
    x_min = min(box1[0], box2[0])
    y_min = min(box1[1], box2[1])
    x_max = max(box1[2], box2[2])
    y_max = max(box1[3], box2[3])
    return [x_min, y_min, x_max, y_max]


def expand_box(box, img_w, img_h, ratio=0.2):
    """等比例扩大box，保持中心不变"""
    x_min, y_min, x_max, y_max = box
    w = x_max - x_min
    h = y_max - y_min
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    new_w = w * (1 + ratio)
    new_h = h * (1 + ratio)

    x_min = max(0, cx - new_w / 2)
    y_min = max(0, cy - new_h / 2)
    x_max = min(img_w, cx + new_w / 2)
    y_max = min(img_h, cy + new_h / 2)
    return [x_min, y_min, x_max, y_max]


def is_contained(inner, outer):
    """判断 inner 框是否被 outer 框完全包裹"""
    return (inner[0] >= outer[0] and
            inner[1] >= outer[1] and
            inner[2] <= outer[2] and
            inner[3] <= outer[3])


def copy_images_by_yolo_labels(label_dir, image_dir, output_dir, img_exts=[".jpg", ".png", ".jpeg"]):
    """
    根据 YOLO 标注文件名，拷贝同名图片到指定目录。

    参数：
        label_dir : str  # YOLO txt 文件夹
        image_dir : str  # 图片文件夹
        output_dir : str # 输出图片文件夹
        img_exts : list  # 支持的图片后缀
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有 YOLO txt 文件
    print(os.path.join(label_dir, "*.txt"))
    label_files = glob.glob(os.path.join(label_dir, "*.txt"))

    if not label_files:
        print(f"⚠️ 没有找到任何标注文件: {label_dir}")
        return

    count = 0
    for label_file in label_files:
        base_name = os.path.splitext(os.path.basename(label_file))[0]

        # 遍历图片后缀，寻找同名图片
        found = False
        for ext in img_exts:
            img_path = os.path.join(image_dir, base_name + ext)
            if os.path.exists(img_path):
                shutil.copy(img_path, output_dir)
                found = True
                count += 1
                break

        if not found:
            print(f"⚠️ 找不到对应图片: {base_name} in {image_dir}")

    print(f"✅ 已完成拷贝 {count} 张图片到 {output_dir}")


def count_yolo_class_ids(label_dir):
    """
    统计 YOLO 标注文件夹:
    {
        "total_images": 图片总数,
        "classes": {
            class_id: {
                "count": 该类总实例数,
                "image_count": 出现过该类的图片数量
            }
        }
    }
    """

    class_counts = defaultdict(int)  # 每类总实例数
    class_images = defaultdict(int)  # 每类出现的图片数
    total_images = 0  # 总图片数量

    if not os.path.exists(label_dir):
        print(f"⚠️ 文件夹不存在: {label_dir}")
        return {
            "total_images": 0,
            "classes": {}
        }

    for file_name in os.listdir(label_dir):
        if not file_name.endswith(".txt"):
            continue

        total_images += 1
        label_path = os.path.join(label_dir, file_name)

        # 用 set 防止同一图片里同一类重复计数
        classes_in_image = set()

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                cls_id = int(parts[0])
                class_counts[cls_id] += 1
                classes_in_image.add(cls_id)

        # 更新每个类的图片计数
        for cls in classes_in_image:
            class_images[cls] += 1

    # 组织返回格式
    result = {
        "total_images": total_images,
        "classes": {}
    }

    for cls_id in sorted(class_counts.keys()):
        result["classes"][cls_id] = {
            "count": class_counts[cls_id],
            "image_count": class_images[cls_id]
        }

    return result


def save_dataset_class_counts_json(dataset_name, counts, json_path):
    """
    将统计结果按数据集名称存入 JSON 文件，不覆盖已有数据。
    counts 结构例如:
    {
        "total_images": 1200,
        "classes": {
            0: {"count": 5432, "image_count": 812},
            1: {"count": 2132, "image_count": 430}
        }
    }
    """

    # 读取已有 JSON
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            try:
                all_data = json.load(f)
            except json.JSONDecodeError:
                all_data = {}
    else:
        all_data = {}

    # ★ JSON 的字典 key 必须是字符串，所以要转换 class_id
    classes_fixed = {
        str(cls_id): {
            "count": v["count"],
            "image_count": v["image_count"]
        }
        for cls_id, v in counts["classes"].items()
    }

    # 整理要保存的结构
    save_data = {
        "total_images": counts["total_images"],
        "classes": classes_fixed
    }

    # 更新指定数据集内容
    all_data[dataset_name] = save_data

    # 写回文件
    with open(json_path, 'w') as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)

    print(f"✅ 已保存 {dataset_name} 统计结果到 {json_path}")


def copy_yolo_dataset(
        src_img_dir,
        src_label_dir,
        dst_img_dir,
        dst_label_dir,
        img_exts=(".jpg", ".jpeg", ".png", ".bmp"),
        require_label=True
):
    """
    将指定路径下的图片与 YOLO 标签文件复制到目标路径

    参数:
        src_img_dir    源图片文件夹
        src_label_dir  源 YOLO 标签文件夹
        dst_img_dir    目标图片文件夹
        dst_label_dir  目标标签文件夹
        img_exts       支持的图片后缀
        require_label  是否要求图片必须有标签文件，True=没有标签则跳过图片
    """
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)

    # 获取所有图片
    img_files = []
    for ext in img_exts:
        img_files.extend(glob.glob(os.path.join(src_img_dir, f"*{ext}")))

    print(f"共找到图片: {len(img_files)} 张")

    copied_count = 0

    for img_path in img_files:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(src_label_dir, base + ".txt")

        # 如果要求必须有标注文件
        if require_label and not os.path.exists(label_path):
            print(f"跳过图片（无YOLO标签）: {img_path}")
            continue

        # 若无标注文件，但无需强制，可只复制图片
        dst_img = os.path.join(dst_img_dir, os.path.basename(img_path))
        shutil.copy(img_path, dst_img)

        if os.path.exists(label_path):
            dst_label = os.path.join(dst_label_dir, base + ".txt")
            shutil.copy(label_path, dst_label)

        copied_count += 1

    print(f"\n完成！成功复制 {copied_count} 组图片与标签。")


def replace_yolo_class_id(label_dir, new_class_id):
    """
    批量修改 YOLO 标注文件夹中所有行的 class_id，并覆盖写回原文件。

    :param label_dir: YOLO 标注文件目录
    :param new_class_id: 新的 class_id（int）
    """

    if not os.path.exists(label_dir):
        print(f"❌ 路径不存在: {label_dir}")
        return

    txt_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    if not txt_files:
        print("⚠️ 该目录下没有 .txt YOLO 标注文件")
        return

    print(f"📌 正在修改 {len(txt_files)} 个 YOLO 标注文件的 class_id …")

    for file_name in txt_files:
        file_path = os.path.join(label_dir, file_name)

        # 读取原文件
        with open(file_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            # 修改 class_id
            parts[0] = str(new_class_id)

            new_lines.append(" ".join(parts) + "\n")

        # 覆盖写回
        with open(file_path, "w") as f:
            f.writelines(new_lines)

    print(f"✅ 完成！已将目录 {label_dir} 中所有标注的 class_id 修改为 {new_class_id}")

def create_director_for_yolo_train_databse(database_source_path="/home/chenkejing/database/AITotal_ProjectDatabase/carpetDatabaseProgrem"):

    os.makedirs(os.path.join(database_source_path, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(database_source_path, "images", "test"), exist_ok=True)
    os.makedirs(os.path.join(database_source_path, "images", "val"), exist_ok=True)

    os.makedirs(os.path.join(database_source_path, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(database_source_path, "labels", "test"), exist_ok=True)
    os.makedirs(os.path.join(database_source_path, "labels", "val"), exist_ok=True)
