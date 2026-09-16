import os
import shutil
import sys
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config
from tqdm import tqdm


def download_file_worker(s3_args):
    """单个文件下载任务 worker"""
    bucket, s3_key, local_path, config = s3_args

    # 每个线程使用独立的 client 实例
    s3_client = boto3.client("s3", config=config)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    s3_client.download_file(bucket, s3_key, local_path)

    return s3_key


def scan_local_files(local_dir: str):
    """
    扫描本地目录下的所有文件。

    返回：
        set[str]
        文件相对于 local_dir 的相对路径
    """
    print(f"正在扫描本地目录: {local_dir} ...")

    local_files = set()

    if not os.path.exists(local_dir):
        print("本地目录不存在，将创建该目录。")
        os.makedirs(local_dir, exist_ok=True)
        return local_files

    for root, _, files in os.walk(local_dir):
        for filename in files:
            full_path = os.path.join(root, filename)

            # 转换为相对于 local_dir 的路径
            relative_path = os.path.relpath(full_path, local_dir)

            # 统一使用 /
            relative_path = relative_path.replace(os.sep, "/")

            local_files.add(relative_path)

    print(f"本地扫描完成，共找到 {len(local_files)} 个文件。")

    return local_files


def fast_s3_incremental_download(s3_uri: str, local_dir: str, max_workers: int = 50):
    """
    S3 → 本地增量下载。

    只下载：
        S3 中存在，但本地不存在的文件。

    :param s3_uri:
        例如：
        s3://bucket-name/path/to/dir/

    :param local_dir:
        本地保存目录

    :param max_workers:
        并行下载线程数
    """

    if not s3_uri.startswith("s3://"):
        raise ValueError("S3 URI 必须以 's3://' 开头")

    # ============================================================
    # 1. 解析 S3 URI
    # ============================================================

    s3_path = s3_uri[5:]

    parts = s3_path.split("/", 1)

    bucket_name = parts[0]
    s3_prefix = parts[1] if len(parts) > 1 else ""

    # 确保 prefix 不以 / 开头
    s3_prefix = s3_prefix.lstrip("/")

    # ============================================================
    # 2. boto3 配置
    # ============================================================

    botocore_config = Config(
        max_pool_connections=max_workers,
        retries={"max_attempts": 5, "mode": "standard"},
    )

    s3_client = boto3.client("s3", config=botocore_config)

    # ============================================================
    # 3. 扫描本地文件
    # ============================================================

    local_files = scan_local_files(local_dir)

    # ============================================================
    # 4. 扫描 S3 文件
    # ============================================================

    print(f"\n正在扫描 S3 文件列表:")
    print(f"{s3_uri}")

    paginator = s3_client.get_paginator("list_objects_v2")

    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)

    # S3 文件列表
    s3_files = []

    for page in pages:

        if "Contents" not in page:
            continue

        for obj in page["Contents"]:

            s3_key = obj["Key"]

            # 跳过目录对象
            if s3_key.endswith("/"):
                continue

            # ====================================================
            # 计算相对于 S3_TARGET_URI 的路径
            # ====================================================

            relative_path = os.path.relpath(s3_key, s3_prefix)

            relative_path = relative_path.replace(os.sep, "/")

            s3_files.append((s3_key, relative_path))

    print(f"S3 扫描完成，共找到 {len(s3_files)} 个文件。")

    # ============================================================
    # 5. 找出 S3 有、本地没有的文件
    # ============================================================

    download_files = []

    for s3_key, relative_path in s3_files:

        if relative_path not in local_files:
            local_file_path = os.path.join(local_dir, relative_path)

            download_files.append(
                (bucket_name, s3_key, local_file_path, botocore_config)
            )

    # ============================================================
    # 6. 打印同步统计
    # ============================================================

    total_s3_files = len(s3_files)
    total_local_files = len(local_files)
    total_download_files = len(download_files)

    print("\n" + "=" * 70)
    print("增量同步检查完成")
    print("=" * 70)

    print(f"S3 文件数量：      {total_s3_files}")
    print(f"本地文件数量：      {total_local_files}")
    print(f"需要下载文件数量：  {total_download_files}")

    print("=" * 70)

    # ============================================================
    # 7. 没有新增文件
    # ============================================================

    if total_download_files == 0:
        print("\n本地数据已经是最新的。")
        print("没有需要下载的文件。")

        return

    # ============================================================
    # 8. 开始多线程下载
    # ============================================================

    print(f"\n发现 {total_download_files} 个本地不存在的文件，" f"开始使用 {max_workers} 个线程下载...\n")

    success_count = 0
    failed_count = 0

    failed_files = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = [
            executor.submit(download_file_worker, task) for task in download_files
        ]

        with tqdm(total=total_download_files, desc="增量下载进度", unit="file") as pbar:

            for future in as_completed(futures):

                try:

                    s3_key = future.result()

                    success_count += 1

                    pbar.update(1)

                except Exception as e:

                    failed_count += 1

                    failed_files.append(str(e))

                    print(f"\n下载文件失败: {e}", file=sys.stderr)

    # ============================================================
    # 9. 最终结果
    # ============================================================

    print("\n" + "=" * 70)
    print("增量下载完成")
    print("=" * 70)

    print(f"S3 文件总数：      {total_s3_files}")
    print(f"本地原有文件：      {total_local_files}")
    print(f"需要下载文件：      {total_download_files}")
    print(f"成功下载：          {success_count}")
    print(f"下载失败：          {failed_count}")

    print(f"本地保存路径：      " f"{os.path.abspath(local_dir)}")

    print("=" * 70)


def extract_and_delete_archives(local_dir: str):
    """
    递归扫描 local_dir 下的所有压缩文件。

    解压规则：
        xxx.zip
        ↓
        xxx/
            ...

    解压成功后删除原始压缩文件。

    支持：
        .zip
        .tar
        .tar.gz
        .tgz
        .tar.bz2
        .tbz2
        .tar.xz
        .txz
    """

    print("\n" + "=" * 70)
    print("开始检查本地压缩文件")
    print("=" * 70)

    archive_extensions = (
        ".zip",
        ".tar",
        ".tar.gz",
        ".tgz",
        ".tar.bz2",
        ".tbz2",
        ".tar.xz",
        ".txz",
    )

    archive_files = []

    # ============================================================
    # 1. 扫描 LOCAL_SAVE_DIR 下所有压缩文件
    # ============================================================

    for root, _, files in os.walk(local_dir):

        for filename in files:

            if filename.lower().endswith(archive_extensions):
                archive_path = os.path.join(root, filename)

                archive_files.append(archive_path)

    print(f"发现 {len(archive_files)} 个压缩文件。")

    if not archive_files:
        print("没有需要解压的压缩文件。")
        return

    success_count = 0
    failed_count = 0

    # ============================================================
    # 2. 逐个处理压缩文件
    # ============================================================

    for archive_path in archive_files:

        try:

            archive_dir = os.path.dirname(archive_path)

            filename = os.path.basename(archive_path)

            # ====================================================
            # 3. 根据压缩文件名生成解压目录
            # ====================================================

            lower_filename = filename.lower()

            if lower_filename.endswith(".tar.gz"):
                folder_name = filename[:-7]

            elif lower_filename.endswith(".tar.bz2"):
                folder_name = filename[:-8]

            elif lower_filename.endswith(".tar.xz"):
                folder_name = filename[:-7]

            elif lower_filename.endswith(".tbz2"):
                folder_name = filename[:-5]

            elif lower_filename.endswith(".txz"):
                folder_name = filename[:-4]

            elif lower_filename.endswith(".tgz"):
                folder_name = filename[:-4]

            elif lower_filename.endswith(".zip"):
                folder_name = filename[:-4]

            elif lower_filename.endswith(".tar"):
                folder_name = filename[:-4]

            else:
                print(f"不支持的压缩格式：{archive_path}")
                failed_count += 1
                continue

            # ====================================================
            # 4. 创建同名解压目录
            # ====================================================

            extract_dir = os.path.join(archive_dir, folder_name)

            os.makedirs(extract_dir, exist_ok=True)

            print("\n" + "-" * 70)
            print(f"压缩文件：{archive_path}")
            print(f"解压目录：{extract_dir}")

            # ====================================================
            # 5. ZIP
            # ====================================================

            if lower_filename.endswith(".zip"):

                with zipfile.ZipFile(archive_path, "r") as zip_ref:

                    zip_ref.extractall(extract_dir)

            # ====================================================
            # 6. TAR 系列
            # ====================================================

            elif tarfile.is_tarfile(archive_path):

                with tarfile.open(archive_path, "r:*") as tar:

                    tar.extractall(extract_dir)

            else:

                print(f"无法识别的压缩文件：{archive_path}")

                failed_count += 1
                continue

            # ====================================================
            # 7. 解压成功
            # ====================================================

            os.remove(archive_path)

            success_count += 1

            print(f"解压成功：{extract_dir}")

            print(f"已删除原压缩文件：{archive_path}")

        except Exception as e:

            failed_count += 1

            print(f"\n解压失败：{archive_path}")

            print(f"错误信息：{e}")

            print("原始压缩文件已保留。")

    # ============================================================
    # 8. 最终统计
    # ============================================================

    print("\n" + "=" * 70)
    print("压缩文件处理完成")
    print("=" * 70)

    print(f"发现压缩文件：      {len(archive_files)}")

    print(f"成功解压并删除：    {success_count}")

    print(f"解压失败：          {failed_count}")

    print("=" * 70)


def flatten_images_labels(dataset_dir: str):
    """
    将数据集目录中的 images / labels 多级目录拍平。

    原始：

        dataset_dir/
        ├── images/
        │   ├── train/
        │   │   ├── xxx.jpg
        │   │   └── yyy.jpg
        │   └── val/
        │       └── zzz.jpg
        │
        └── labels/
            ├── train/
            │   ├── xxx.txt
            │   └── yyy.txt
            └── val/
                └── zzz.txt

    整理后：

        dataset_dir/
        ├── images/
        │   ├── xxx.jpg
        │   ├── yyy.jpg
        │   └── zzz.jpg
        │
        └── labels/
            ├── xxx.txt
            ├── yyy.txt
            └── zzz.txt

    注意：
        只处理 images 和 labels 目录。
        其他文件不处理。
    """

    print("\n" + "=" * 70)
    print(f"开始拍平 images / labels：{dataset_dir}")
    print("=" * 70)

    images_dir = os.path.join(dataset_dir, "images")

    labels_dir = os.path.join(dataset_dir, "labels")

    # ============================================================
    # 检查目录
    # ============================================================

    if not os.path.isdir(images_dir):
        print(f"[WARNING] images 目录不存在：" f"{images_dir}")

        return

    if not os.path.isdir(labels_dir):
        print(f"[WARNING] labels 目录不存在：" f"{labels_dir}")

        return

    # ============================================================
    # 支持的图片格式
    # ============================================================

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # ============================================================
    # 统计
    # ============================================================

    image_files = []
    label_files = []

    # ============================================================
    # 1. 遍历 images 下所有子目录
    # ============================================================

    for root, _, files in os.walk(images_dir):

        for filename in files:

            file_path = os.path.join(root, filename)

            extension = os.path.splitext(filename)[1].lower()

            if extension in image_extensions:

                # 排除 images 根目录本身的文件
                # 因为这些已经是目标位置
                if os.path.abspath(root) == os.path.abspath(images_dir):
                    continue

                image_files.append(file_path)

    # ============================================================
    # 2. 遍历 labels 下所有子目录
    # ============================================================

    for root, _, files in os.walk(labels_dir):

        for filename in files:

            if not filename.lower().endswith(".txt"):
                continue

            file_path = os.path.join(root, filename)

            # 排除 labels 根目录本身的文件
            if os.path.abspath(root) == os.path.abspath(labels_dir):
                continue

            label_files.append(file_path)

    # ============================================================
    # 3. 输出扫描结果
    # ============================================================

    print(f"发现嵌套图片：{len(image_files)}")

    print(f"发现嵌套 TXT：{len(label_files)}")

    # ============================================================
    # 4. 移动图片到 images 根目录
    # ============================================================

    moved_images = 0
    skipped_images = 0
    image_conflicts = 0

    for image_path in tqdm(image_files, desc="整理 images", unit="file"):

        filename = os.path.basename(image_path)

        target_path = os.path.join(images_dir, filename)

        # --------------------------------------------------------
        # 目标已经存在
        # --------------------------------------------------------

        if os.path.exists(target_path):
            image_conflicts += 1

            print(f"\n[WARNING] " f"images 中存在同名文件：" f"{filename}")

            print(f"  源文件：{image_path}")

            print(f"  目标文件：{target_path}")

            skipped_images += 1

            continue

        # --------------------------------------------------------
        # 移动
        # --------------------------------------------------------

        shutil.move(image_path, target_path)

        moved_images += 1

    # ============================================================
    # 5. 移动 TXT 到 labels 根目录
    # ============================================================

    moved_labels = 0
    skipped_labels = 0
    label_conflicts = 0

    for label_path in tqdm(label_files, desc="整理 labels", unit="file"):

        filename = os.path.basename(label_path)

        target_path = os.path.join(labels_dir, filename)

        # --------------------------------------------------------
        # 目标已经存在
        # --------------------------------------------------------

        if os.path.exists(target_path):
            label_conflicts += 1

            print(f"\n[WARNING] " f"labels 中存在同名文件：" f"{filename}")

            print(f"  源文件：{label_path}")

            print(f"  目标文件：{target_path}")

            skipped_labels += 1

            continue

        # --------------------------------------------------------
        # 移动
        # --------------------------------------------------------

        shutil.move(label_path, target_path)

        moved_labels += 1

    # ============================================================
    # 6. 删除已经变空的子目录
    # ============================================================

    # 从深层目录开始删除
    for root, dirs, files in os.walk(images_dir, topdown=False):

        # 不删除 images 根目录
        if os.path.abspath(root) == os.path.abspath(images_dir):
            continue

        # 如果目录已经为空
        if not os.listdir(root):
            os.rmdir(root)

    for root, dirs, files in os.walk(labels_dir, topdown=False):

        # 不删除 labels 根目录
        if os.path.abspath(root) == os.path.abspath(labels_dir):
            continue

        if not os.listdir(root):
            os.rmdir(root)

    # ============================================================
    # 7. 最终统计
    # ============================================================

    print("\n" + "-" * 70)
    print("images / labels 拍平完成")
    print("-" * 70)

    print(f"发现嵌套图片：      {len(image_files)}")

    print(f"移动图片：          {moved_images}")

    print(f"跳过图片：          {skipped_images}")

    print(f"图片同名冲突：      {image_conflicts}")

    print(f"发现嵌套 TXT：       {len(label_files)}")

    print(f"移动 TXT：           {moved_labels}")

    print(f"跳过 TXT：           {skipped_labels}")

    print(f"TXT 同名冲突：       {label_conflicts}")

    print(f"\nimages：{images_dir}")

    print(f"labels：{labels_dir}")

    print("-" * 70)


def flatten_all_extracted_datasets(local_dir: str):
    """
    遍历 local_dir 下所有已经解压的数据集目录，
    对每个数据集执行 images / labels 拍平。
    """

    print("\n" + "=" * 70)
    print("开始拍平所有解压数据集")
    print("=" * 70)

    dataset_dirs = []

    for filename in os.listdir(local_dir):

        dataset_dir = os.path.join(local_dir, filename)

        if not os.path.isdir(dataset_dir):
            continue

        # 只处理包含 images / labels 的目录
        images_dir = os.path.join(dataset_dir, "images")

        labels_dir = os.path.join(dataset_dir, "labels")

        if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
            dataset_dirs.append(dataset_dir)

    print(f"发现 {len(dataset_dirs)} 个数据集目录。")

    success_count = 0
    failed_count = 0

    for dataset_dir in dataset_dirs:

        try:

            flatten_images_labels(dataset_dir)

            success_count += 1

        except Exception as e:

            failed_count += 1

            print(f"\n[ERROR] " f"处理失败：{dataset_dir}")

            print(f"错误信息：{e}")

    print("\n" + "=" * 70)
    print("所有数据集拍平完成")
    print("=" * 70)

    print(f"数据集总数：{len(dataset_dirs)}")

    print(f"成功：{success_count}")

    print(f"失败：{failed_count}")

    print("=" * 70)


if __name__ == "__main__":
    # ============================================================
    # 自定义配置
    # ============================================================

    S3_TARGET_URI = "s3://robot-ai-platform/datasets/carpet_detection/annotations/cvat/"

    LOCAL_SAVE_DIR = "/data/database/AITotal_SegmentDatabase/cvat_dataset/carpet_detect/date_carpet_20260915"

    MAX_WORKERS = 64

    # ============================================================
    # 执行增量下载
    # ============================================================

    fast_s3_incremental_download(
        S3_TARGET_URI,
        LOCAL_SAVE_DIR,
        max_workers=MAX_WORKERS
    )

    # ============================================================
    # 2. 下载完成后，解压所有压缩文件
    # ============================================================

    extract_and_delete_archives(
        LOCAL_SAVE_DIR
    )

    # ============================================================
    # 3. images / labels 拍平
    # ============================================================

    flatten_all_extracted_datasets(LOCAL_SAVE_DIR)
