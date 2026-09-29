import os
import shutil
import sys
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config
from tqdm import tqdm

# ================================================================
# 压缩文件相关工具函数
# ================================================================

ARCHIVE_EXTENSIONS = (
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".txz",
)


def get_archive_base_name(filename: str):
    """
    获取压缩文件对应的数据集基础名称。

    支持：

        xxx.zip
        xxx.zip.93a7A263

        xxx.tar
        xxx.tar.xxxxx

        xxx.tar.gz
        xxx.tar.gz.xxxxx

        xxx.tgz
        xxx.tgz.xxxxx

        xxx.tar.bz2
        xxx.tar.bz2.xxxxx

        xxx.tar.xz
        xxx.tar.xz.xxxxx

    最终返回：

        xxx

    例如：

        carpet_detection_xxx.zip.93a7A263
        ↓
        carpet_detection_xxx
    """

    filename_lower = filename.lower()

    # 按最长扩展名优先判断
    archive_markers = (
        ".tar.bz2",
        ".tar.gz",
        ".tar.xz",
        ".tbz2",
        ".txz",
        ".tgz",
        ".zip",
        ".tar",
    )

    for marker in archive_markers:

        index = filename_lower.find(marker)

        if index == -1:
            continue

        # ========================================================
        # 确保 marker 是文件名结尾或者后面是随机后缀
        #
        # 例如：
        #
        # xxx.zip
        # xxx.zip.93a7A263
        #
        # 都合法
        # ========================================================

        suffix_start = index + len(marker)

        if suffix_start < len(filename):

            suffix = filename[suffix_start:]

            # 如果 marker 后面不是 "." 开头，
            # 说明不是合法的压缩文件后缀形式
            if not suffix.startswith("."):
                continue

        # ========================================================
        # 去掉压缩扩展名
        # ========================================================

        return filename[:index]

    return None


def get_archive_type(filename: str):
    """
    判断压缩文件类型。

    返回：

        zip
        tar
        None
    """

    filename_lower = filename.lower()

    archive_markers = (
        ".tar.bz2",
        ".tar.gz",
        ".tar.xz",
        ".tbz2",
        ".txz",
        ".tgz",
        ".zip",
        ".tar",
    )

    for marker in archive_markers:

        index = filename_lower.find(marker)

        if index == -1:
            continue

        suffix_start = index + len(marker)

        if suffix_start < len(filename):

            suffix = filename[suffix_start:]

            if not suffix.startswith("."):
                continue

        if marker == ".zip":
            return "zip"

        return "tar"

    return None


def get_archive_folder_name(filename: str):
    """
    根据压缩文件名获取解压目录名称。

    例如：

        xxx.zip
        -> xxx

        xxx.zip.93a7A263
        -> xxx

        xxx.tar.gz
        -> xxx

        xxx.tar.gz.123456
        -> xxx
    """

    return get_archive_base_name(filename)


# ================================================================
# S3 下载
# ================================================================

def download_file_worker(s3_args):
    """单个文件下载任务 worker"""

    bucket, s3_key, local_path, config = s3_args

    # 每个线程使用独立的 client 实例
    s3_client = boto3.client(
        "s3",
        config=config
    )

    os.makedirs(
        os.path.dirname(local_path),
        exist_ok=True
    )

    s3_client.download_file(
        bucket,
        s3_key,
        local_path
    )

    return s3_key


def scan_local_files(local_dir: str):
    """
    扫描本地目录下的所有文件。

    返回：
        set[str]

        文件相对于 local_dir 的相对路径
    """

    print(
        f"正在扫描本地目录: "
        f"{local_dir} ..."
    )

    local_files = set()

    if not os.path.exists(local_dir):
        print(
            "本地目录不存在，将创建该目录。"
        )

        os.makedirs(
            local_dir,
            exist_ok=True
        )

        return local_files

    for root, _, files in os.walk(local_dir):

        for filename in files:
            full_path = os.path.join(
                root,
                filename
            )

            # 转换为相对于 local_dir 的路径
            relative_path = os.path.relpath(
                full_path,
                local_dir
            )

            # 统一使用 /
            relative_path = relative_path.replace(
                os.sep,
                "/"
            )

            local_files.add(
                relative_path
            )

    print(
        f"本地扫描完成，共找到 "
        f"{len(local_files)} 个文件。"
    )

    return local_files


def scan_sibling_files(local_dir: str):
    """
    扫描 LOCAL_SAVE_DIR 上一级目录下所有子目录。

    重点：

    当前目录：

        carpet_detect/date_carpet_20260928/

    上一级：

        carpet_detect/

    历史目录：

        carpet_detect/
        ├── date_carpet_20260915/
        │   ├── dataset_A/
        │   ├── dataset_B/
        │   └── dataset_C/
        │
        └── date_carpet_20260928/

    本函数会递归扫描上一级目录下所有目录，
    记录所有目录名称。

    例如：

        dataset_A
        dataset_B
        dataset_C

    用于判断 S3 压缩文件是否已经被下载并解压过。
    """

    parent_dir = os.path.dirname(
        os.path.abspath(local_dir)
    )

    print(
        f"正在扫描历史数据目录: "
        f"{parent_dir} ..."
    )

    sibling_dataset_dirs = set()

    if not os.path.isdir(parent_dir):
        print(
            f"历史数据目录不存在："
            f"{parent_dir}"
        )

        return sibling_dataset_dirs

    # ============================================================
    # 遍历上一层目录下所有子目录
    # ============================================================

    for root, dirs, files in os.walk(parent_dir):

        for dirname in dirs:

            full_dir = os.path.join(
                root,
                dirname
            )

            # 当前目录本身不需要作为历史数据
            if os.path.abspath(full_dir) == os.path.abspath(local_dir):
                continue

            sibling_dataset_dirs.add(
                dirname
            )

    print(
        f"历史数据目录扫描完成，共找到 "
        f"{len(sibling_dataset_dirs)} 个目录名称。"
    )

    return sibling_dataset_dirs


def fast_s3_incremental_download(
        s3_uri: str,
        local_dir: str,
        max_workers: int = 50
):
    """
    S3 → 本地增量下载。

    原有规则：

        S3 中存在
        +
        当前 LOCAL_SAVE_DIR 不存在
        =
        下载

    新增规则：

        如果 S3 对象是压缩文件：

            xxx.zip
            xxx.zip.xxxxx
            xxx.tar.gz
            xxx.tar.gz.xxxxx

        则提取数据集基础名称：

            xxx

        如果 LOCAL_SAVE_DIR 上一级目录的历史目录中
        已经存在同名目录：

            xxx/

        则跳过下载。
    """

    if not s3_uri.startswith("s3://"):
        raise ValueError(
            "S3 URI 必须以 's3://' 开头"
        )

    # ============================================================
    # 1. 解析 S3 URI
    # ============================================================

    s3_path = s3_uri[5:]

    parts = s3_path.split(
        "/",
        1
    )

    bucket_name = parts[0]

    s3_prefix = (
        parts[1]
        if len(parts) > 1
        else ""
    )

    # 确保 prefix 不以 / 开头
    s3_prefix = s3_prefix.lstrip("/")

    # ============================================================
    # 2. boto3 配置
    # ============================================================

    botocore_config = Config(
        max_pool_connections=max_workers,
        retries={
            "max_attempts": 5,
            "mode": "standard"
        },
    )

    s3_client = boto3.client(
        "s3",
        config=botocore_config
    )

    # ============================================================
    # 3. 扫描当前本地文件
    # ============================================================

    local_files = scan_local_files(
        local_dir
    )

    # ============================================================
    # 3.1 扫描历史数据集目录
    # ============================================================

    sibling_files = scan_sibling_files(
        local_dir
    )

    # ============================================================
    # 4. 扫描 S3 文件
    # ============================================================

    print(
        "\n正在扫描 S3 文件列表:"
    )

    print(
        s3_uri
    )

    paginator = s3_client.get_paginator(
        "list_objects_v2"
    )

    pages = paginator.paginate(
        Bucket=bucket_name,
        Prefix=s3_prefix
    )

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

            relative_path = os.path.relpath(
                s3_key,
                s3_prefix
            )

            relative_path = relative_path.replace(
                os.sep,
                "/"
            )

            s3_files.append(
                (
                    s3_key,
                    relative_path
                )
            )

    print(
        f"S3 扫描完成，共找到 "
        f"{len(s3_files)} 个文件。"
    )

    # ============================================================
    # 5. 找出需要下载的文件
    # ============================================================

    download_files = []

    # 因历史目录已经存在而跳过的压缩包数量
    duplicate_archive_count = 0

    # ============================================================
    # 遍历 S3 文件
    # ============================================================

    for s3_key, relative_path in s3_files:

        # ========================================================
        # 5.1 当前 LOCAL_SAVE_DIR 已经存在
        # ========================================================

        if relative_path in local_files:
            continue

        # ========================================================
        # 5.2 获取文件名
        # ========================================================

        filename = os.path.basename(
            relative_path
        )

        # ========================================================
        # 5.3 判断是不是压缩文件
        # ========================================================

        archive_base_name = get_archive_base_name(
            filename
        )

        # ========================================================
        # 5.4 如果是压缩文件
        #
        # 例如：
        #
        # S3:
        #
        # xxx.zip.93a7A263
        #
        # 转换：
        #
        # xxx
        #
        # 检查：
        #
        # 历史目录是否存在：
        #
        # xxx/
        #
        # ========================================================

        if archive_base_name is not None:

            if archive_base_name in sibling_files:
                duplicate_archive_count += 1

                print(
                    f"跳过历史已存在的数据集："
                    f"{filename}"
                )

                continue

        # ========================================================
        # 5.5 确定需要下载
        # ========================================================

        local_file_path = os.path.join(
            local_dir,
            relative_path
        )

        download_files.append(
            (
                bucket_name,
                s3_key,
                local_file_path,
                botocore_config
            )
        )

    # ============================================================
    # 6. 打印同步统计
    # ============================================================

    total_s3_files = len(
        s3_files
    )

    total_local_files = len(
        local_files
    )

    total_download_files = len(
        download_files
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "增量同步检查完成"
    )

    print(
        "=" * 70
    )

    print(
        f"S3 文件数量：              "
        f"{total_s3_files}"
    )

    print(
        f"当前本地文件数量：          "
        f"{total_local_files}"
    )

    print(
        f"历史目录数量：              "
        f"{len(sibling_files)}"
    )

    print(
        f"历史同名数据集跳过：        "
        f"{duplicate_archive_count}"
    )

    print(
        f"需要下载文件数量：          "
        f"{total_download_files}"
    )

    print(
        "=" * 70
    )

    # ============================================================
    # 7. 没有新增文件
    # ============================================================

    if total_download_files == 0:
        print(
            "\n本地数据已经是最新的。"
        )

        print(
            "没有需要下载的文件。"
        )

        return

    # ============================================================
    # 8. 开始多线程下载
    # ============================================================

    print(
        f"\n发现 {total_download_files} "
        f"个本地不存在的文件，"
        f"开始使用 {max_workers} 个线程下载...\n"
    )

    success_count = 0

    failed_count = 0

    failed_files = []

    with ThreadPoolExecutor(
            max_workers=max_workers
    ) as executor:

        futures = [
            executor.submit(
                download_file_worker,
                task
            )
            for task in download_files
        ]

        with tqdm(
                total=total_download_files,
                desc="增量下载进度",
                unit="file"
        ) as pbar:

            for future in as_completed(
                    futures
            ):

                try:

                    s3_key = future.result()

                    success_count += 1

                    pbar.update(1)

                except Exception as e:

                    failed_count += 1

                    failed_files.append(
                        str(e)
                    )

                    print(
                        f"\n下载文件失败: {e}",
                        file=sys.stderr
                    )

    # ============================================================
    # 9. 最终结果
    # ============================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "增量下载完成"
    )

    print(
        "=" * 70
    )

    print(
        f"S3 文件总数：      "
        f"{total_s3_files}"
    )

    print(
        f"本地原有文件：      "
        f"{total_local_files}"
    )

    print(
        f"需要下载文件：      "
        f"{total_download_files}"
    )

    print(
        f"成功下载：          "
        f"{success_count}"
    )

    print(
        f"下载失败：          "
        f"{failed_count}"
    )

    print(
        f"本地保存路径：      "
        f"{os.path.abspath(local_dir)}"
    )

    print(
        "=" * 70
    )


# ================================================================
# 解压
# ================================================================

def extract_and_delete_archives(local_dir: str):
    """
    递归扫描 local_dir 下的所有压缩文件。

    支持：

        xxx.zip
        xxx.zip.xxxxx

        xxx.tar
        xxx.tar.xxxxx

        xxx.tar.gz
        xxx.tar.gz.xxxxx

        xxx.tgz
        xxx.tgz.xxxxx

        xxx.tar.bz2
        xxx.tar.bz2.xxxxx

        xxx.tar.xz
        xxx.tar.xz.xxxxx

        xxx.tbz2
        xxx.tbz2.xxxxx

        xxx.txz
        xxx.txz.xxxxx

    解压成功后删除原始压缩文件。
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "开始检查本地压缩文件"
    )

    print(
        "=" * 70
    )

    archive_files = []

    # ============================================================
    # 1. 扫描 LOCAL_SAVE_DIR 下所有压缩文件
    # ============================================================

    for root, _, files in os.walk(
            local_dir
    ):

        for filename in files:

            archive_type = get_archive_type(
                filename
            )

            if archive_type is None:
                continue

            archive_path = os.path.join(
                root,
                filename
            )

            archive_files.append(
                archive_path
            )

    print(
        f"发现 {len(archive_files)} "
        f"个压缩文件。"
    )

    if not archive_files:
        print(
            "没有需要解压的压缩文件。"
        )

        return

    success_count = 0

    failed_count = 0

    # ============================================================
    # 2. 逐个处理压缩文件
    # ============================================================

    for archive_path in archive_files:

        try:

            archive_dir = os.path.dirname(
                archive_path
            )

            filename = os.path.basename(
                archive_path
            )

            # ====================================================
            # 3. 判断压缩文件类型
            # ====================================================

            archive_type = get_archive_type(
                filename
            )

            folder_name = get_archive_folder_name(
                filename
            )

            if archive_type is None:
                print(
                    f"不支持的压缩格式："
                    f"{archive_path}"
                )

                failed_count += 1

                continue

            if folder_name is None:
                print(
                    f"无法获取解压目录名称："
                    f"{archive_path}"
                )

                failed_count += 1

                continue

            # ====================================================
            # 4. 创建同名解压目录
            # ====================================================

            extract_dir = os.path.join(
                archive_dir,
                folder_name
            )

            os.makedirs(
                extract_dir,
                exist_ok=True
            )

            print(
                "\n" + "-" * 70
            )

            print(
                f"压缩文件："
                f"{archive_path}"
            )

            print(
                f"解压目录："
                f"{extract_dir}"
            )

            # ====================================================
            # 5. ZIP
            # ====================================================

            if archive_type == "zip":

                with zipfile.ZipFile(
                        archive_path,
                        "r"
                ) as zip_ref:

                    zip_ref.extractall(
                        extract_dir
                    )

            # ====================================================
            # 6. TAR 系列
            # ====================================================

            elif archive_type == "tar":

                with tarfile.open(
                        archive_path,
                        "r:*"
                ) as tar:

                    tar.extractall(
                        extract_dir
                    )

            else:

                print(
                    f"无法识别的压缩文件："
                    f"{archive_path}"
                )

                failed_count += 1

                continue

            # ====================================================
            # 7. 解压成功
            # ====================================================

            os.remove(
                archive_path
            )

            success_count += 1

            print(
                f"解压成功："
                f"{extract_dir}"
            )

            print(
                f"已删除原压缩文件："
                f"{archive_path}"
            )

        except Exception as e:

            failed_count += 1

            print(
                f"\n解压失败："
                f"{archive_path}"
            )

            print(
                f"错误信息：{e}"
            )

            print(
                "原始压缩文件已保留。"
            )

    # ============================================================
    # 8. 最终统计
    # ============================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "压缩文件处理完成"
    )

    print(
        "=" * 70
    )

    print(
        f"发现压缩文件：      "
        f"{len(archive_files)}"
    )

    print(
        f"成功解压并删除：    "
        f"{success_count}"
    )

    print(
        f"解压失败：          "
        f"{failed_count}"
    )

    print(
        "=" * 70
    )


# ================================================================
# images / labels 拍平
# ================================================================

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

    print(
        "\n" + "=" * 70
    )

    print(
        f"开始拍平 images / labels："
        f"{dataset_dir}"
    )

    print(
        "=" * 70
    )

    images_dir = os.path.join(
        dataset_dir,
        "images"
    )

    labels_dir = os.path.join(
        dataset_dir,
        "labels"
    )

    # ============================================================
    # 检查目录
    # ============================================================

    if not os.path.isdir(
            images_dir
    ):
        print(
            f"[WARNING] images 目录不存在："
            f"{images_dir}"
        )

        return

    if not os.path.isdir(
            labels_dir
    ):
        print(
            f"[WARNING] labels 目录不存在："
            f"{labels_dir}"
        )

        return

    # ============================================================
    # 支持的图片格式
    # ============================================================

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    # ============================================================
    # 统计
    # ============================================================

    image_files = []

    label_files = []

    # ============================================================
    # 1. 遍历 images 下所有子目录
    # ============================================================

    for root, _, files in os.walk(
            images_dir
    ):

        for filename in files:

            file_path = os.path.join(
                root,
                filename
            )

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension in image_extensions:

                # 排除 images 根目录本身的文件
                if (
                        os.path.abspath(root)
                        == os.path.abspath(images_dir)
                ):
                    continue

                image_files.append(
                    file_path
                )

    # ============================================================
    # 2. 遍历 labels 下所有子目录
    # ============================================================

    for root, _, files in os.walk(
            labels_dir
    ):

        for filename in files:

            if not filename.lower().endswith(
                    ".txt"
            ):
                continue

            file_path = os.path.join(
                root,
                filename
            )

            # 排除 labels 根目录本身的文件
            if (
                    os.path.abspath(root)
                    == os.path.abspath(labels_dir)
            ):
                continue

            label_files.append(
                file_path
            )

    # ============================================================
    # 3. 输出扫描结果
    # ============================================================

    print(
        f"发现嵌套图片："
        f"{len(image_files)}"
    )

    print(
        f"发现嵌套 TXT："
        f"{len(label_files)}"
    )

    # ============================================================
    # 4. 移动图片到 images 根目录
    # ============================================================

    moved_images = 0

    skipped_images = 0

    image_conflicts = 0

    for image_path in tqdm(
            image_files,
            desc="整理 images",
            unit="file"
    ):

        filename = os.path.basename(
            image_path
        )

        target_path = os.path.join(
            images_dir,
            filename
        )

        if os.path.exists(
                target_path
        ):
            image_conflicts += 1

            print(
                f"\n[WARNING] "
                f"images 中存在同名文件："
                f"{filename}"
            )

            print(
                f"  源文件："
                f"{image_path}"
            )

            print(
                f"  目标文件："
                f"{target_path}"
            )

            skipped_images += 1

            continue

        shutil.move(
            image_path,
            target_path
        )

        moved_images += 1

    # ============================================================
    # 5. 移动 TXT 到 labels 根目录
    # ============================================================

    moved_labels = 0

    skipped_labels = 0

    label_conflicts = 0

    for label_path in tqdm(
            label_files,
            desc="整理 labels",
            unit="file"
    ):

        filename = os.path.basename(
            label_path
        )

        target_path = os.path.join(
            labels_dir,
            filename
        )

        if os.path.exists(
                target_path
        ):
            label_conflicts += 1

            print(
                f"\n[WARNING] "
                f"labels 中存在同名文件："
                f"{filename}"
            )

            print(
                f"  源文件："
                f"{label_path}"
            )

            print(
                f"  目标文件："
                f"{target_path}"
            )

            skipped_labels += 1

            continue

        shutil.move(
            label_path,
            target_path
        )

        moved_labels += 1

    # ============================================================
    # 6. 删除已经变空的子目录
    # ============================================================

    for root, dirs, files in os.walk(
            images_dir,
            topdown=False
    ):

        if (
                os.path.abspath(root)
                == os.path.abspath(images_dir)
        ):
            continue

        if not os.listdir(root):
            os.rmdir(root)

    for root, dirs, files in os.walk(
            labels_dir,
            topdown=False
    ):

        if (
                os.path.abspath(root)
                == os.path.abspath(labels_dir)
        ):
            continue

        if not os.listdir(root):
            os.rmdir(root)

    # ============================================================
    # 7. 最终统计
    # ============================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "images / labels 拍平完成"
    )

    print(
        "-" * 70
    )

    print(
        f"发现嵌套图片：      "
        f"{len(image_files)}"
    )

    print(
        f"移动图片：          "
        f"{moved_images}"
    )

    print(
        f"跳过图片：          "
        f"{skipped_images}"
    )

    print(
        f"图片同名冲突：      "
        f"{image_conflicts}"
    )

    print(
        f"发现嵌套 TXT：       "
        f"{len(label_files)}"
    )

    print(
        f"移动 TXT：           "
        f"{moved_labels}"
    )

    print(
        f"跳过 TXT：           "
        f"{skipped_labels}"
    )

    print(
        f"TXT 同名冲突：       "
        f"{label_conflicts}"
    )

    print(
        f"\nimages："
        f"{images_dir}"
    )

    print(
        f"labels："
        f"{labels_dir}"
    )

    print(
        "-" * 70
    )


# ================================================================
# 拍平所有数据集
# ================================================================

def flatten_all_extracted_datasets(
        local_dir: str
):
    """
    遍历 local_dir 下所有已经解压的数据集目录，
    对每个数据集执行 images / labels 拍平。
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "开始拍平所有解压数据集"
    )

    print(
        "=" * 70
    )

    dataset_dirs = []

    for filename in os.listdir(
            local_dir
    ):

        dataset_dir = os.path.join(
            local_dir,
            filename
        )

        if not os.path.isdir(
                dataset_dir
        ):
            continue

        # 只处理包含 images / labels 的目录
        images_dir = os.path.join(
            dataset_dir,
            "images"
        )

        labels_dir = os.path.join(
            dataset_dir,
            "labels"
        )

        if (
                os.path.isdir(images_dir)
                and os.path.isdir(labels_dir)
        ):
            dataset_dirs.append(
                dataset_dir
            )

    print(
        f"发现 {len(dataset_dirs)} "
        f"个数据集目录。"
    )

    success_count = 0

    failed_count = 0

    for dataset_dir in dataset_dirs:

        try:

            flatten_images_labels(
                dataset_dir
            )

            success_count += 1

        except Exception as e:

            failed_count += 1

            print(
                f"\n[ERROR] "
                f"处理失败："
                f"{dataset_dir}"
            )

            print(
                f"错误信息："
                f"{e}"
            )

    print(
        "\n" + "=" * 70
    )

    print(
        "所有数据集拍平完成"
    )

    print(
        "=" * 70
    )

    print(
        f"数据集总数："
        f"{len(dataset_dirs)}"
    )

    print(
        f"成功："
        f"{success_count}"
    )

    print(
        f"失败："
        f"{failed_count}"
    )

    print(
        "=" * 70
    )


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    # ============================================================
    # 自定义配置
    # ============================================================

    S3_TARGET_URI = (
        "s3://robot-ai-platform/"
        "datasets/carpet_detection/"
        "annotations/cvat/"
    )

    LOCAL_SAVE_DIR = (
        "/data/database/"
        "AITotal_SegmentDatabase/"
        "cvat_dataset/"
        "carpet_detect/"
        "date_carpet_20260928"
    )

    MAX_WORKERS = 64

    # ============================================================
    # 1. 执行增量下载
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

    flatten_all_extracted_datasets(
        LOCAL_SAVE_DIR
    )
