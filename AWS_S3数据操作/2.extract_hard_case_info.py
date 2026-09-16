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

# 2. 需要拼接的前缀路径字符串 (例如服务器/远程/相对路径字符串)
base_prefix = (
    "/workspace/data/AITotal_SegmentDatabase/carpetDatabaseSegment/images/train"
)

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
