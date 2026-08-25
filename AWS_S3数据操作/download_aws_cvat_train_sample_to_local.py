import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config
from tqdm import tqdm

# ============================================================
# S3 配置
# ============================================================

S3_BUCKET = "robot-ai-platform"

# 实际目录：
#
# s3://robot-ai-platform/datasets/carpet_detection/annotations/cvat/
# s3://robot-ai-platform/datasets/wire_detection/annotations/cvat/
#
S3_DATASET_ROOT = "datasets"

# ============================================================
# 并发配置
# ============================================================

# 同时下载多少个 ZIP
MAX_WORKERS = 4

# S3 Client 连接池
MAX_POOL_CONNECTIONS = 16

# ============================================================
# tqdm 输出锁
# ============================================================

tqdm_lock = threading.Lock()


# ============================================================
# 从文件名解析项目名称
# ============================================================


def parse_project_name(filename):
    """
    从文件名中解析项目名称。

    例如：

    carpet_detection_UT-A10XCNA00014A006_20260819_carpet_hard_case_mask

    ->
    carpet_detection


    wire_detection_UT-A10XCNA00815T006_20260811_wire on floor_hard_case_mask

    ->
    wire_detection
    """

    filename = Path(filename).name

    # 去掉 .zip
    if filename.lower().endswith(".zip"):
        filename = filename[:-4]

    match = re.match(r"^(.+?_detection)_", filename, re.IGNORECASE)

    if not match:
        raise ValueError(
            "\n无法从文件名中解析项目名称：\n"
            f"{filename}\n\n"
            "正确格式应该类似：\n"
            "carpet_detection_UT-A10XCNA00014A006_20260819_carpet_hard_case_mask"
        )

    return match.group(1)


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
# 查找 ZIP 文件
# ============================================================


def find_zip_file(s3_client, bucket, project_name, zip_filename):
    """
    在对应项目的 CVAT 目录中查找 ZIP 文件。

    例如：

    project_name:
        carpet_detection

    自动搜索：

    s3://robot-ai-platform/
        datasets/
        carpet_detection/
        annotations/
        cvat/
    """

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


def download_one_zip(file_name, local_dir):
    """
    下载一个 ZIP 文件。

    参数：

    file_name:
        不需要 .zip

    local_dir:
        本地保存目录
    """

    try:

        # ----------------------------------------------------
        # 清理文件名
        # ----------------------------------------------------

        file_name = Path(file_name).name

        if file_name.lower().endswith(".zip"):
            file_name = file_name[:-4]

        # ----------------------------------------------------
        # 解析项目
        # ----------------------------------------------------

        project_name = parse_project_name(file_name)

        # ----------------------------------------------------
        # 自动补 .zip
        # ----------------------------------------------------

        zip_filename = file_name + ".zip"

        # ----------------------------------------------------
        # S3 Prefix
        # ----------------------------------------------------

        s3_prefix = f"{S3_DATASET_ROOT}/" f"{project_name}/" f"annotations/" f"cvat/"

        print()
        print(f"[开始] {zip_filename}")

        print(f"       项目：{project_name}")

        print(f"       S3：" f"s3://{S3_BUCKET}/{s3_prefix}")

        # ----------------------------------------------------
        # 每个线程使用自己的 S3 Client
        # ----------------------------------------------------

        s3_client = create_s3_client(max_pool_connections=MAX_POOL_CONNECTIONS)

        # ----------------------------------------------------
        # 搜索文件
        # ----------------------------------------------------

        s3_key = find_zip_file(
            s3_client=s3_client,
            bucket=S3_BUCKET,
            project_name=project_name,
            zip_filename=zip_filename,
        )

        if s3_key is None:
            print()
            print(f"[失败] 找不到：{zip_filename}")

            return {"file_name": file_name, "success": False, "reason": "S3文件不存在"}

        # ----------------------------------------------------
        # 本地路径
        # ----------------------------------------------------

        local_dir = Path(local_dir)

        local_dir.mkdir(parents=True, exist_ok=True)

        local_path = local_dir / zip_filename

        # ----------------------------------------------------
        # 如果本地已经存在
        # ----------------------------------------------------

        if local_path.exists():
            print()
            print(f"[跳过] 本地已存在：" f"{local_path}")

            return {"file_name": file_name, "success": True, "skipped": True}

        # ----------------------------------------------------
        # 获取文件大小
        # ----------------------------------------------------

        response = s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)

        total_size = response["ContentLength"]

        # ----------------------------------------------------
        # 下载
        # ----------------------------------------------------

        print(f"[下载] {zip_filename}")

        print(f"       大小：" f"{sizeof_fmt(total_size)}")

        print(f"       S3：" f"s3://{S3_BUCKET}/{s3_key}")

        # ----------------------------------------------------
        # tqdm
        # ----------------------------------------------------

        with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=zip_filename[:35],
                position=0,
                leave=True,
        ) as pbar:

            def progress_callback(bytes_amount):

                pbar.update(bytes_amount)

            try:

                s3_client.download_file(
                    Bucket=S3_BUCKET,
                    Key=s3_key,
                    Filename=str(local_path),
                    Callback=progress_callback,
                )

            except Exception:

                # 删除不完整文件
                if local_path.exists():
                    local_path.unlink()

                raise

        print(f"[完成] {zip_filename}")

        return {
            "file_name": file_name,
            "success": True,
            "skipped": False,
            "local_path": str(local_path),
        }

    except Exception as e:

        print()
        print(f"[失败] {file_name}")

        print(f"       {e}")

        return {"file_name": file_name, "success": False, "reason": str(e)}


# ============================================================
# 批量下载
# ============================================================


def download_multiple_zips(file_names, local_dir, max_workers=4):
    """
    批量并行下载多个 ZIP。

    file_names:
        文件名列表

    例如：

    [
        "carpet_detection_xxx",
        "carpet_detection_yyy",
        "wire_detection_xxx"
    ]
    """

    total_files = len(file_names)

    if total_files == 0:
        print("没有需要下载的文件。")

        return

    print()
    print("=" * 80)
    print("CVAT ZIP 批量下载")
    print("=" * 80)

    print(f"文件数量：{total_files}")

    print(f"并发线程：{max_workers}")

    print(f"保存目录：{local_dir}")

    print("=" * 80)

    results = []

    # --------------------------------------------------------
    # ThreadPool
    # --------------------------------------------------------

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        future_map = {}

        for file_name in file_names:
            future = executor.submit(download_one_zip, file_name, local_dir)

            future_map[future] = file_name

        # ----------------------------------------------------
        # 等待任务完成
        # ----------------------------------------------------

        for future in as_completed(future_map):

            file_name = future_map[future]

            try:

                result = future.result()

                results.append(result)

            except Exception as e:

                print()
                print(f"[异常] {file_name}")

                print(e)

                results.append(
                    {"file_name": file_name, "success": False, "reason": str(e)}
                )

    # ========================================================
    # 统计
    # ========================================================

    success_count = 0
    skipped_count = 0
    failed_count = 0

    for result in results:

        if not result["success"]:

            failed_count += 1

        elif result.get("skipped", False):

            skipped_count += 1

        else:

            success_count += 1

    # ========================================================
    # 输出结果
    # ========================================================

    print()
    print()
    print("=" * 80)
    print("批量下载完成")
    print("=" * 80)

    print(f"总文件数：{total_files}")

    print(f"成功下载：{success_count}")

    print(f"已存在跳过：{skipped_count}")

    print(f"下载失败：{failed_count}")

    print("=" * 80)

    # --------------------------------------------------------
    # 失败列表
    # --------------------------------------------------------

    failed_results = [r for r in results if not r["success"]]

    if failed_results:

        print()
        print("失败文件：")

        for result in failed_results:
            print(f"  {result['file_name']}")

            print(f"      原因：" f"{result.get('reason', '')}")

    print()


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    # ========================================================
    # 文件名列表
    #
    # 注意：
    # 不需要写 .zip
    # ========================================================

    FILE_NAMES = [
        # ----------------------------------------------------
        # carpet
        # ----------------------------------------------------
        (
            "carpet_detection_UT-A10XCNA00014A006_20260819_carpet_hard_case_mask"
        ),
        (
            "carpet_detection_A10-25-YD-005002-test_20260730_carpet_hard_case_mask"
        ),
        # ----------------------------------------------------
        # wire
        # ----------------------------------------------------
        (
            "wire_detection_UT-A10XCNA00815T006_20260811_wire on floor_hard_case_mask"
        ),
    ]

    # ========================================================
    # 本地保存目录
    # ========================================================

    LOCAL_SAVE_DIR = "/home/chenkejing/Downloads/cvat_zip"

    # ========================================================
    # 开始批量下载
    # ========================================================

    download_multiple_zips(
        file_names=FILE_NAMES, local_dir=LOCAL_SAVE_DIR, max_workers=MAX_WORKERS
    )
