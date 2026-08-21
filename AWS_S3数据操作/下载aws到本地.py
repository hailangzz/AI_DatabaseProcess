import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config
from tqdm import tqdm


def download_file_worker(s3_args):
    """单个文件下载任务 worker"""
    bucket, s3_key, local_path, config = s3_args
    # 每个线程使用独立的 client 实例以保证线程安全
    s3_client = boto3.client('s3', config=config)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_client.download_file(bucket, s3_key, local_path)
    return s3_key


def fast_s3_download(s3_uri: str, local_dir: str, max_workers: int = 50):
    """
    多线程并行下载 S3 文件夹到本地
    :param s3_uri: 例如 's3://bucket-name/path/to/dir/'
    :param local_dir: 本地保存路径
    :param max_workers: 并行线程数（针对小文件推荐 32 - 64）
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError("S3 URI 必须以 's3://' 开头")

    s3_path = s3_uri[5:]
    parts = s3_path.split("/", 1)
    bucket_name = parts[0]
    s3_prefix = parts[1] if len(parts) > 1 else ""

    # 配置独立的 S3 Client 参数
    botocore_config = Config(
        max_pool_connections=max_workers,
        retries={'max_attempts': 5, 'mode': 'standard'}
    )
    s3_client = boto3.client('s3', config=botocore_config)

    print(f"正在扫描 S3 文件列表: {s3_uri} ...")
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)

    tasks = []
    for page in pages:
        if 'Contents' not in page:
            continue
        for obj in page['Contents']:
            s3_key = obj['Key']
            if s3_key.endswith('/'):  # 跳过纯目录对象
                continue

            # 计算本地保存路径
            relative_path = os.path.relpath(s3_key, s3_prefix)
            local_file_path = os.path.join(local_dir, relative_path)

            # 组装任务参数
            tasks.append((bucket_name, s3_key, local_file_path, botocore_config))

    total_files = len(tasks)
    print(f"扫描完成，共找到 {total_files} 个文件，开始使用 {max_workers} 个线程并行下载...\n")

    if total_files == 0:
        print("未找到任何可下载的文件。")
        return

    # 使用线程池并发下载
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_file_worker, task) for task in tasks]

        # 使用 tqdm 显示并行下载进度条
        with tqdm(total=total_files, desc="下载进度", unit="file") as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                    pbar.update(1)
                except Exception as e:
                    print(f"\n下载文件失败: {e}", file=sys.stderr)

    print(f"\n全部下载完成！共 {total_files} 个文件已保存至: {os.path.abspath(local_dir)}")


if __name__ == "__main__":
    # ================= 自定义配置 =================
    S3_TARGET_URI = "s3://robot-ai-platform/datasets/wire_detection/annotations/sam3/yolo/UT-A10XCNA00815T006/20260811/null/"  # 替换为你的 S3 路径
    LOCAL_SAVE_DIR = "/home/chenkejing/Downloads/WireSegmentProject/downloaded_dataset_null"  # 替换为本地保存路径
    MAX_WORKERS = 64  # 如果大部分是几百 KB 到几 MB 的小文件，设为 64 或 128 速度极快
    # ==============================================

    fast_s3_download(S3_TARGET_URI, LOCAL_SAVE_DIR, max_workers=MAX_WORKERS)
