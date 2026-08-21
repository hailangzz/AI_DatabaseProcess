import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# ============================================================
# 创建 S3 Client
# ============================================================

s3 = boto3.client("s3")


# ============================================================
# 获取本地所有文件
# ============================================================


def get_all_files(local_dir):
    files = []

    for root, dirs, filenames in os.walk(local_dir):

        for filename in filenames:

            file_path = os.path.join(root, filename)

            if os.path.isfile(file_path):
                files.append(file_path)

    return files


# ============================================================
# 获取 S3 Key
# ============================================================


def get_s3_key(local_file, local_dir, s3_prefix):
    # 获取相对路径
    relative_path = os.path.relpath(local_file, local_dir)

    # Windows 路径转换成 /
    relative_path = relative_path.replace("\\", "/")

    # 拼接 S3 Key
    s3_key = os.path.join(s3_prefix, relative_path)

    # 再次确保使用 /
    s3_key = s3_key.replace("\\", "/")

    return s3_key


# ============================================================
# 判断 S3 文件是否已经存在
# ============================================================


def s3_file_exists(bucket, key, local_file):
    try:

        response = s3.head_object(Bucket=bucket, Key=key)

        # 如果大小相同，认为已经上传
        local_size = os.path.getsize(local_file)

        s3_size = response["ContentLength"]

        return local_size == s3_size

    except ClientError as e:

        error_code = e.response["Error"]["Code"]

        if error_code in ("404", "NoSuchKey"):
            return False

        raise


# ============================================================
# 上传单个文件
# ============================================================


def upload_file(local_file):
    s3_key = get_s3_key(local_file, LOCAL_DIR, S3_PREFIX)

    # --------------------------------------------------------
    # 判断是否已经存在
    # --------------------------------------------------------

    try:

        if s3_file_exists(S3_BUCKET, s3_key, local_file):
            return {"status": "skip", "file": local_file}

    except Exception as e:

        return {"status": "error", "file": local_file, "error": str(e)}

    # --------------------------------------------------------
    # 上传
    # --------------------------------------------------------

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            s3.upload_file(local_file, S3_BUCKET, s3_key)

            return {"status": "success", "file": local_file}

        except (BotoCoreError, ClientError) as e:

            if attempt == MAX_RETRIES:
                return {"status": "error", "file": local_file, "error": str(e)}

    return {"status": "error", "file": local_file, "error": "unknown error"}


# ============================================================
# 主函数
# ============================================================


def sync_to_s3():
    print("=" * 60)

    print("S3 Sync")

    print("=" * 60)

    print(f"Local : {LOCAL_DIR}")

    print(f"S3    : s3://{S3_BUCKET}/{S3_PREFIX}")

    print(f"Threads: {MAX_WORKERS}")

    print("=" * 60)

    # --------------------------------------------------------
    # 扫描文件
    # --------------------------------------------------------

    files = get_all_files(LOCAL_DIR)

    total = len(files)

    print(f"发现文件：{total}")

    if total == 0:
        print("没有发现文件")

        return

    # --------------------------------------------------------
    # 多线程上传
    # --------------------------------------------------------

    success = 0

    skipped = 0

    failed = 0

    failed_files = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = [executor.submit(upload_file, file) for file in files]

        for index, future in enumerate(as_completed(futures), 1):

            result = future.result()

            status = result["status"]

            file = result["file"]

            if status == "success":

                success += 1

                print(f"[{index}/{total}] " f"上传成功: {file}")

            elif status == "skip":

                skipped += 1

                print(f"[{index}/{total}] " f"跳过: {file}")

            else:

                failed += 1

                failed_files.append(result)

                print(f"[{index}/{total}] " f"上传失败: {file}")

                print(f"    错误: " f"{result['error']}")

    # --------------------------------------------------------
    # 结果
    # --------------------------------------------------------

    print()

    print("=" * 60)

    print("同步完成")

    print("=" * 60)

    print(f"总文件数 : {total}")

    print(f"上传成功 : {success}")

    print(f"跳过文件 : {skipped}")

    print(f"上传失败 : {failed}")

    print("=" * 60)

    # --------------------------------------------------------
    # 输出失败文件
    # --------------------------------------------------------

    if failed_files:

        print()

        print("失败文件：")

        for item in failed_files:
            print(item["file"])

            print(f"    {item['error']}")


# ============================================================
# 配置
# ============================================================

# 本地目录 (设备SN目录的上级目录)
LOCAL_DIR = (r"/data/database/carpetDatabase")

# S3 Bucket
S3_BUCKET = "robot-ai-platform"

# S3 目标目录
S3_PREFIX = "datasets/carpet_detection" + "/source/images"
# S3_PREFIX = "datasets/wire_detection" + "/source/images"

# 线程数量
MAX_WORKERS = 20

# 失败重试次数
MAX_RETRIES = 3

# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    sync_to_s3()
