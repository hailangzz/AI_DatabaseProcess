#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OSS -> AWS S3 数据迁移工具

功能：
1. 递归读取 OSS 指定 prefix 下的所有 Object
2. 直接从 OSS 流式读取
3. 直接上传到 AWS S3
4. 不经过本地磁盘
5. 多线程并发
6. S3 已存在且大小一致时跳过
7. 失败自动重试
8. 输出成功 / 跳过 / 失败统计

数据流：

Alibaba Cloud OSS
        |
        | get_object
        v
    Python stream
        |
        | upload_fileobj
        v
AWS S3
"""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import oss2
from botocore.exceptions import ClientError
import os

# ============================================================
# 全局统计
# ============================================================

lock = threading.Lock()

total_count = 0
success_count = 0
skip_count = 0
failed_count = 0

failed_objects = []


# ============================================================
# Credential
# ============================================================

def get_oss_bucket():
    """
    从环境变量读取 OSS AK/SK。
    """

    # access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
    # access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")
    access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
    access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")

    if not access_key_id or not access_key_secret:
        raise RuntimeError(
            "OSS credentials not found.\n"
            "Please set:\n"
            "  OSS_ACCESS_KEY_ID\n"
            "  OSS_ACCESS_KEY_SECRET"
        )

    auth = oss2.Auth(
        access_key_id,
        access_key_secret
    )

    bucket = oss2.Bucket(
        auth,
        OSS_ENDPOINT,
        OSS_BUCKET_NAME
    )

    return bucket


def get_s3_client():
    """
    使用 boto3 默认 credential provider chain。

    支持：
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - ~/.aws/credentials
    - IAM Role
    - EC2 Instance Profile
    等。
    """

    return boto3.client(
        "s3",
        region_name=AWS_REGION
    )


# ============================================================
# S3 检查
# ============================================================

def get_s3_object_size(s3_client, key):
    """
    获取 S3 Object 大小。

    返回：
        int  : Object size
        None : Object 不存在
    """

    try:
        response = s3_client.head_object(
            Bucket=AWS_BUCKET_NAME,
            Key=key
        )

        return response["ContentLength"]

    except ClientError as e:

        error_code = e.response.get("Error", {}).get("Code")

        if error_code in ("404", "NoSuchKey", "NotFound"):
            return None

        raise


# ============================================================
# 单个 Object
# ============================================================

def copy_object(oss_bucket, s3_client, oss_key):
    """
    OSS -> S3

    不写本地文件，直接流式复制。
    """

    global success_count
    global skip_count
    global failed_count

    s3_key = f"{AWS_DEST_PREFIX}{oss_key}"

    # --------------------------------------------------------
    # 获取 OSS Object 信息
    # --------------------------------------------------------

    try:
        oss_info = oss_bucket.head_object(oss_key)

        oss_size = oss_info.content_length

    except Exception as e:

        with lock:
            failed_count += 1
            failed_objects.append(
                (oss_key, f"OSS HEAD failed: {e}")
            )

        return False

    # --------------------------------------------------------
    # 检查 S3
    # --------------------------------------------------------

    if not OVERWRITE:

        try:

            s3_size = get_s3_object_size(
                s3_client,
                s3_key
            )

            if s3_size is not None and s3_size == oss_size:
                with lock:
                    skip_count += 1

                print(
                    f"[SKIP] {oss_key}"
                )

                return True

        except Exception as e:

            print(
                f"[WARN] S3 HEAD failed: "
                f"{s3_key} | {e}"
            )

    # --------------------------------------------------------
    # 上传
    # --------------------------------------------------------

    for attempt in range(MAX_RETRIES):

        try:

            print(
                f"[UPLOAD] {oss_key} "
                f"-> s3://{AWS_BUCKET_NAME}/{s3_key}"
            )

            # 从 OSS 获取流
            oss_object = oss_bucket.get_object(
                oss_key
            )

            # 直接把 OSS response body 作为文件对象
            s3_client.upload_fileobj(
                oss_object,
                AWS_BUCKET_NAME,
                s3_key
            )

            with lock:
                success_count += 1

            print(
                f"[SUCCESS] {oss_key}"
            )

            return True

        except Exception as e:

            if attempt < MAX_RETRIES - 1:

                sleep_time = 2 ** attempt

                print(
                    f"[RETRY {attempt + 1}/{MAX_RETRIES}] "
                    f"{oss_key} | {e} | "
                    f"sleep {sleep_time}s"
                )

                time.sleep(sleep_time)

            else:

                with lock:
                    failed_count += 1
                    failed_objects.append(
                        (oss_key, str(e))
                    )

                print(
                    f"[FAILED] {oss_key} | {e}"
                )

                return False


# ============================================================
# 主程序
# ============================================================

def main():
    global total_count

    print("=" * 70)
    print("OSS -> AWS S3 Migration")
    print("=" * 70)

    print(f"OSS bucket : {OSS_BUCKET_NAME}")
    print(f"OSS prefix : {OSS_PREFIX}")
    print(f"S3 bucket  : {AWS_BUCKET_NAME}")
    print(f"S3 prefix  : {AWS_DEST_PREFIX}")
    print(f"Workers    : {MAX_WORKERS}")
    print(f"Retries    : {MAX_RETRIES}")
    print("=" * 70)

    # --------------------------------------------------------
    # 初始化
    # --------------------------------------------------------

    try:

        oss_bucket = get_oss_bucket()

        # 每个线程都可以使用 boto3 client。
        # boto3 client 官方推荐线程安全使用。
        s3_client = get_s3_client()

    except Exception as e:

        print(f"[FATAL] Initialization failed: {e}")

        sys.exit(1)

    # --------------------------------------------------------
    # 列出 OSS Object
    # --------------------------------------------------------

    print("\nScanning OSS objects...")

    objects = []

    try:

        for obj in oss2.ObjectIterator(
                oss_bucket,
                prefix=OSS_PREFIX
        ):

            # 跳过目录 marker
            if obj.key.endswith("/"):
                continue

            objects.append(
                (obj.key, obj.size)
            )

    except Exception as e:

        print(
            f"[FATAL] Failed to list OSS objects: {e}"
        )

        sys.exit(1)

    total_count = len(objects)

    print(
        f"Found {total_count} objects."
    )

    if total_count == 0:
        print("Nothing to copy.")
        return

    # --------------------------------------------------------
    # 多线程
    # --------------------------------------------------------

    start_time = time.time()

    with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
    ) as executor:

        futures = []

        for oss_key, _ in objects:
            future = executor.submit(
                copy_object,
                oss_bucket,
                s3_client,
                oss_key
            )

            futures.append(future)

        completed = 0

        for future in as_completed(futures):

            completed += 1

            try:
                future.result()

            except Exception as e:

                print(
                    f"[THREAD ERROR] {e}"
                )

            print(
                f"\nProgress: "
                f"{completed}/{total_count} "
                f"({completed / total_count * 100:.2f}%)"
            )

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    elapsed = time.time() - start_time

    print("\n")
    print("=" * 70)
    print("Migration finished")
    print("=" * 70)

    print(f"Total   : {total_count}")
    print(f"Success : {success_count}")
    print(f"Skip    : {skip_count}")
    print(f"Failed  : {failed_count}")

    print(
        f"Elapsed : {elapsed:.2f}s"
    )

    if elapsed > 0:
        print(
            f"Speed   : "
            f"{total_count / elapsed:.2f} objects/s"
        )

    # --------------------------------------------------------
    # 失败列表
    # --------------------------------------------------------

    if failed_objects:

        failed_file = "oss_to_s3_failed.txt"

        with open(
                failed_file,
                "w",
                encoding="utf-8"
        ) as f:

            for key, error in failed_objects:
                f.write(
                    f"{key}\t{error}\n"
                )

        print(
            f"\nFailed objects saved to:"
            f" {failed_file}"
        )

    print("=" * 70)


# ============================================================
# 配置
# ============================================================

# ---------- OSS ----------
OSS_ENDPOINT = "https://oss-cn-shenzhen.aliyuncs.com"
OSS_BUCKET_NAME = "robot-ai-platform"

# 要复制的 OSS 目录
# 例如：
# datasets/carpet_detection/source/images/
OSS_PREFIX = "datasets/carpet_detection/source/images/UT-A10XCNA00014A007/"

# ---------- AWS ----------
AWS_BUCKET_NAME = "robot-ai-platform"

# AWS Region
AWS_REGION = "ap-northeast-1"

# 如果希望 OSS 的 key 原样复制到 AWS，则保持为空字符串。
AWS_DEST_PREFIX = ""

# ---------- 并发 ----------
MAX_WORKERS = 24

# ---------- 重试 ----------
MAX_RETRIES = 3

# ---------- 是否覆盖 ----------
# False：
#   S3 已存在且大小一致 -> 跳过
#
# True：
#   即使存在也重新上传
OVERWRITE = False

if __name__ == "__main__":
    main()
