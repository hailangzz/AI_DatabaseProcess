import os
import re
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config
from tqdm import tqdm

# ============================================================
# S3 配置
# ============================================================

S3_BUCKET = "robot-ai-platform"

# 实际：
#
# s3://robot-ai-platform/datasets/carpet_detection/annotations/cvat/
# s3://robot-ai-platform/datasets/wire_detection/annotations/cvat/
#
S3_DATASET_ROOT = "datasets"

# ============================================================
# 并发配置
# ============================================================

# ZIP 下载 / ZIP 处理并发数
MAX_WORKERS = 4

# S3 connection pool
MAX_POOL_CONNECTIONS = 16

# ============================================================
# 任务类型映射
# ============================================================

PROJECT_TO_TASK = {
    "carpet_detection": "carpet_detect",
    "wire_detection": "wire_detect",
    "liquid_detection": "liquid_detect",
    "plasticbag_detection": "plasticbag_detect",
}

# ============================================================
# 支持的图像格式
# ============================================================

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# 从 ZIP 文件名解析项目
# ============================================================


def parse_project_name(filename):
    """
    例如：

    carpet_detection_UT-A10XCNA00014A006_20260819_carpet_hard_case_mask

    返回：

    carpet_detection
    """

    filename = Path(filename).name

    if filename.lower().endswith(".zip"):
        filename = filename[:-4]

    match = re.match(r"^(.+?_detection)_", filename, re.IGNORECASE)

    if not match:
        raise ValueError("\n无法解析项目名称：\n" f"{filename}\n")

    return match.group(1)


# ============================================================
# 根据项目名称解析任务名称
# ============================================================


def parse_task_name(project_name):
    """
    例如：

    carpet_detection
        ->
    carpet_detect

    wire_detection
        ->
    wire_detect
    """

    # 优先使用显式映射
    if project_name in PROJECT_TO_TASK:
        return PROJECT_TO_TASK[project_name]

    # 没有配置时自动转换
    if project_name.endswith("_detection"):
        return project_name[: -len("_detection")] + "_detect"

    raise ValueError(f"无法解析任务名称：" f"{project_name}")


# ============================================================
# 创建 S3 Client
# ============================================================


def create_s3_client(max_pool_connections=16):
    config = Config(
        max_pool_connections=max_pool_connections,
        retries={"max_attempts": 5, "mode": "standard"},
    )

    return boto3.client("s3", config=config)


# ============================================================
# 查找 ZIP
# ============================================================


def find_zip_file(s3_client, bucket, project_name, zip_filename):
    s3_prefix = f"{S3_DATASET_ROOT}/" f"{project_name}/" f"annotations/" f"cvat/"

    paginator = s3_client.get_paginator("list_objects_v2")

    pages = paginator.paginate(Bucket=bucket, Prefix=s3_prefix)

    for page in pages:

        contents = page.get("Contents", [])

        for obj in contents:

            s3_key = obj["Key"]

            filename = os.path.basename(s3_key)

            if filename == zip_filename:
                return s3_key

    return None


# ============================================================
# 文件大小格式化
# ============================================================


def sizeof_fmt(num):
    for unit in ["B", "KB", "MB", "GB", "TB"]:

        if num < 1024:
            return f"{num:.2f} {unit}"

        num /= 1024

    return f"{num:.2f} PB"


# ============================================================
# 下载单个 ZIP
# ============================================================


def download_one_zip(file_name, local_zip_dir):
    try:

        # ----------------------------------------------------
        # 清理文件名
        # ----------------------------------------------------

        file_name = Path(file_name).name

        if file_name.lower().endswith(".zip"):
            file_name = file_name[:-4]

        # ----------------------------------------------------
        # 项目
        # ----------------------------------------------------

        project_name = parse_project_name(file_name)

        # ----------------------------------------------------
        # 任务
        # ----------------------------------------------------

        task_name = parse_task_name(project_name)

        # ----------------------------------------------------
        # ZIP
        # ----------------------------------------------------

        zip_filename = file_name + ".zip"

        # ----------------------------------------------------
        # S3
        # ----------------------------------------------------

        s3_prefix = f"{S3_DATASET_ROOT}/" f"{project_name}/" f"annotations/" f"cvat/"

        print()
        print(f"[开始] {zip_filename}")

        print(f"       项目：" f"{project_name}")

        print(f"       任务：" f"{task_name}")

        # ----------------------------------------------------
        # S3 Client
        # ----------------------------------------------------

        s3_client = create_s3_client(MAX_POOL_CONNECTIONS)

        # ----------------------------------------------------
        # 搜索
        # ----------------------------------------------------

        s3_key = find_zip_file(
            s3_client=s3_client,
            bucket=S3_BUCKET,
            project_name=project_name,
            zip_filename=zip_filename,
        )

        if s3_key is None:
            print(f"[失败] S3 不存在：" f"{zip_filename}")

            return {"success": False, "file_name": file_name, "reason": "S3文件不存在"}

        # ----------------------------------------------------
        # 本地 ZIP 目录
        # ----------------------------------------------------

        local_zip_dir = Path(local_zip_dir)

        local_zip_dir.mkdir(parents=True, exist_ok=True)

        local_zip_path = local_zip_dir / zip_filename

        # ----------------------------------------------------
        # 已存在
        # ----------------------------------------------------

        if local_zip_path.exists():
            print(f"[跳过] 本地已存在：" f"{local_zip_path}")

            return {
                "success": True,
                "file_name": file_name,
                "zip_path": str(local_zip_path),
                "task_name": task_name,
                "project_name": project_name,
                "skipped": True,
            }

        # ----------------------------------------------------
        # 获取大小
        # ----------------------------------------------------

        response = s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)

        total_size = response["ContentLength"]

        # ----------------------------------------------------
        # 下载
        # ----------------------------------------------------

        print(f"[下载] {zip_filename}")

        print(f"       大小：" f"{sizeof_fmt(total_size)}")

        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=zip_filename[:35],
            leave=True,
        ) as pbar:

            def callback(bytes_amount):

                pbar.update(bytes_amount)

            try:

                s3_client.download_file(
                    Bucket=S3_BUCKET,
                    Key=s3_key,
                    Filename=str(local_zip_path),
                    Callback=callback,
                )

            except Exception:

                if local_zip_path.exists():
                    local_zip_path.unlink()

                raise

        print(f"[完成] {zip_filename}")

        return {
            "success": True,
            "file_name": file_name,
            "zip_path": str(local_zip_path),
            "task_name": task_name,
            "project_name": project_name,
            "skipped": False,
        }

    except Exception as e:

        print()
        print(f"[失败] {file_name}")

        print(f"       {e}")

        return {"success": False, "file_name": file_name, "reason": str(e)}


# ============================================================
# 批量下载
# ============================================================


def download_multiple_zips(file_names, local_zip_dir, max_workers=4):
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        future_map = {}

        for file_name in file_names:
            future = executor.submit(download_one_zip, file_name, local_zip_dir)

            future_map[future] = file_name

        for future in as_completed(future_map):

            file_name = future_map[future]

            try:

                result = future.result()

                results.append(result)

            except Exception as e:

                print(f"[异常] " f"{file_name}")

                print(e)

    return results


# ============================================================
# ZIP 解压
# ============================================================


def extract_zip(zip_path, extract_root):
    zip_path = Path(zip_path)

    extract_root = Path(extract_root)

    extract_root.mkdir(parents=True, exist_ok=True)

    # ZIP 名称作为目录名称
    extract_dir = extract_root / zip_path.stem

    extract_dir.mkdir(parents=True, exist_ok=True)

    print()
    print(f"[解压] {zip_path.name}")

    with zipfile.ZipFile(zip_path, "r") as zf:

        # ----------------------------------------------------
        # ZIP 路径安全检查
        # ----------------------------------------------------

        extract_dir_resolved = extract_dir.resolve()

        for member in zf.infolist():

            target_path = (extract_dir / member.filename).resolve()

            if not str(target_path).startswith(str(extract_dir_resolved)):
                raise RuntimeError("ZIP 中存在非法路径：" f"{member.filename}")

        zf.extractall(extract_dir)

    print(f"[解压完成] " f"{extract_dir}")

    return extract_dir


# ============================================================
# 扫描图片
# ============================================================


def find_images(extract_dir):
    images = []

    for path in Path(extract_dir).rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)

    return images


# ============================================================
# 扫描 TXT 标签
# ============================================================


def find_labels(extract_dir):
    labels = []

    for path in Path(extract_dir).rglob("*.txt"):

        if path.is_file():
            labels.append(path)

    return labels


# ============================================================
# 整理一个 ZIP
# ============================================================


def organize_one_zip(zip_path, extract_root, output_root):
    zip_path = Path(zip_path)

    # --------------------------------------------------------
    # 解析：
    #
    # carpet_detection_xxx
    #
    # ->
    #
    # carpet_detection
    # ->
    #
    # carpet_detect
    # --------------------------------------------------------

    file_name = zip_path.stem

    project_name = parse_project_name(file_name)

    task_name = parse_task_name(project_name)

    # --------------------------------------------------------
    # 最终任务目录
    # --------------------------------------------------------

    task_root = Path(output_root) / task_name

    # --------------------------------------------------------
    # 当前 ZIP 对应的数据目录
    # --------------------------------------------------------

    dataset_root = task_root / file_name

    images_dir = dataset_root / "images"

    labels_dir = dataset_root / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)

    labels_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 解压
    # --------------------------------------------------------

    extract_dir = extract_zip(zip_path, extract_root)

    # --------------------------------------------------------
    # 找图片
    # --------------------------------------------------------

    image_files = find_images(extract_dir)

    # --------------------------------------------------------
    # 找标签
    # --------------------------------------------------------

    label_files = find_labels(extract_dir)

    # --------------------------------------------------------
    # 建立 label 映射
    #
    # xxx.txt
    # ->
    # xxx
    # --------------------------------------------------------

    label_map = {}

    for label_path in label_files:
        stem = label_path.stem

        label_map[stem] = label_path

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    copied_images = 0
    copied_labels = 0
    generated_empty_labels = 0

    image_conflicts = 0
    label_conflicts = 0

    # --------------------------------------------------------
    # 处理图片
    # --------------------------------------------------------

    for image_path in image_files:

        image_name = image_path.name

        image_stem = image_path.stem

        destination_image = images_dir / image_name

        # ----------------------------------------------
        # 图片已经存在
        # ----------------------------------------------

        if destination_image.exists():

            image_conflicts += 1

        else:

            shutil.copy2(image_path, destination_image)

            copied_images += 1

        # ----------------------------------------------
        # 对应 TXT
        # ----------------------------------------------

        destination_label = labels_dir / f"{image_stem}.txt"

        # 已经存在 label
        if destination_label.exists():
            continue

        # ----------------------------------------------
        # ZIP 中存在 TXT
        # ----------------------------------------------

        if image_stem in label_map:

            shutil.copy2(label_map[image_stem], destination_label)

            copied_labels += 1

        # ----------------------------------------------
        # ZIP 中没有 TXT
        #
        # 自动创建空 TXT
        # ----------------------------------------------

        else:

            destination_label.touch()

            generated_empty_labels += 1

    # --------------------------------------------------------
    # 处理那些没有对应图片的 TXT
    #
    # 这里不复制。
    #
    # YOLO 数据集最终以 image 为基准。
    # --------------------------------------------------------

    orphan_labels = 0

    image_stems = {image.stem for image in image_files}

    for label_path in label_files:

        if label_path.stem not in image_stems:
            orphan_labels += 1

    # --------------------------------------------------------
    # 输出
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(f"任务：{task_name}")

    print(f"ZIP：{file_name}")

    print(f"图片：{len(image_files)}")

    print(f"原始 TXT：{len(label_files)}")

    print(f"复制图片：{copied_images}")

    print(f"复制标签：{copied_labels}")

    print(f"自动生成空标签：" f"{generated_empty_labels}")

    print(f"图片冲突：" f"{image_conflicts}")

    print(f"孤立 TXT：" f"{orphan_labels}")

    print(f"输出：" f"{dataset_root}")

    print("=" * 70)

    return {
        "success": True,
        "task_name": task_name,
        "dataset_root": str(dataset_root),
        "images": len(image_files),
        "labels": len(label_files),
        "copied_images": copied_images,
        "copied_labels": copied_labels,
        "empty_labels": generated_empty_labels,
        "image_conflicts": image_conflicts,
        "orphan_labels": orphan_labels,
    }


# ============================================================
# 批量整理 ZIP
# ============================================================


def organize_multiple_zips(zip_paths, extract_root, output_root):
    results = []

    for zip_path in zip_paths:

        try:

            result = organize_one_zip(
                zip_path=zip_path, extract_root=extract_root, output_root=output_root
            )

            results.append(result)

        except Exception as e:

            print()
            print(f"[处理失败] " f"{zip_path}")

            print(f"原因：{e}")

            results.append({"success": False, "zip": str(zip_path), "reason": str(e)})

    # ========================================================
    # 总统计
    # ========================================================

    total_images = sum(r.get("copied_images", 0) for r in results)

    total_labels = sum(r.get("copied_labels", 0) for r in results)

    total_empty_labels = sum(r.get("empty_labels", 0) for r in results)

    print()
    print()
    print("=" * 80)
    print("全部数据整理完成")
    print("=" * 80)

    print(f"处理 ZIP：" f"{len(results)}")

    print(f"新增图片：" f"{total_images}")

    print(f"新增 TXT：" f"{total_labels}")

    print(f"自动生成空 TXT：" f"{total_empty_labels}")

    print(f"最终数据目录：" f"{output_root}")

    print("=" * 80)

    return results


# ============================================================
# main
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # 1. CVAT ZIP 名称
    #
    # 不需要 .zip
    # ========================================================

    FILE_NAMES = [
        # ----------------------------------------------------
        # carpet
        # ----------------------------------------------------
        (
            "carpet_detection_"
            "UT-A10XCNA00014A006_"
            "20260819_"
            "carpet_hard_case_mask"
        ),
        (
            "carpet_detection_"
            "A10-25-YD-005002-test_"
            "20260730_"
            "carpet_hard_case_mask"
        ),
        # ----------------------------------------------------
        # wire
        # ----------------------------------------------------
        (
            "wire_detection_"
            "UT-A10XCNA00815T006_"
            "20260811_"
            "wire on floor_hard_case_mask"
        ),
    ]

    # ========================================================
    # 2. ZIP 保存目录
    # ========================================================

    ZIP_SAVE_DIR = "/home/chenkejing/Downloads/" "cvat_zip"

    # ========================================================
    # 3. ZIP 解压目录
    # ========================================================

    EXTRACT_ROOT = "/home/chenkejing/Downloads/" "cvat_zip/extracted"

    # ========================================================
    # 4. 最终数据集目录
    # ========================================================

    DATASET_OUTPUT_ROOT = "/home/chenkejing/Downloads/" "cvat_dataset"

    # ========================================================
    # 5. 下载 ZIP
    # ========================================================

    download_results = download_multiple_zips(
        file_names=FILE_NAMES, local_zip_dir=ZIP_SAVE_DIR, max_workers=MAX_WORKERS
    )

    # ========================================================
    # 6. 提取成功的 ZIP
    # ========================================================

    zip_paths = []

    for result in download_results:

        if result.get("success", False):
            zip_paths.append(Path(result["zip_path"]))

    # ========================================================
    # 7. 解压 + 整理
    # ========================================================

    organize_multiple_zips(
        zip_paths=zip_paths, extract_root=EXTRACT_ROOT, output_root=DATASET_OUTPUT_ROOT
    )

    print()
    print("全部任务执行完成。")
