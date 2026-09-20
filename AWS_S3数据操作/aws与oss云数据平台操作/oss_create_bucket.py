# -*- coding: utf-8 -*-

import os
import sys

import oss2

# ==================== 配置 ====================

BUCKET_NAME = "robot-ai-platform"

ENDPOINT = "https://oss-cn-shenzhen.aliyuncs.com"

ACCESS_KEY_ID = os.getenv("OSS_ACCESS_KEY_ID")
ACCESS_KEY_SECRET = os.getenv("OSS_ACCESS_KEY_SECRET")


# =================================================


def create_oss_bucket(
        bucket_name: str,
        endpoint: str,
        access_key_id: str,
        access_key_secret: str,
        storage_class=oss2.BUCKET_STORAGE_CLASS_STANDARD,
        acl=oss2.BUCKET_ACL_PRIVATE,
):
    """创建 OSS Bucket"""

    if not access_key_id or not access_key_secret:
        print("错误：没有找到 OSS AccessKey。")
        print("请设置：")
        print("  OSS_ACCESS_KEY_ID")
        print("  OSS_ACCESS_KEY_SECRET")
        sys.exit(1)

    # 1. 创建认证
    auth = oss2.Auth(
        access_key_id,
        access_key_secret
    )

    # 2. 创建 Bucket 对象
    bucket = oss2.Bucket(
        auth,
        endpoint,
        bucket_name
    )

    print(f"开始尝试创建 Bucket: {bucket_name}")
    print(f"Endpoint: {endpoint}")
    print(f"Storage Class: {storage_class}")
    print(f"ACL: {acl}")

    try:

        # 3. 创建 Bucket 配置
        bucket_config = oss2.models.BucketCreateConfig(
            storage_class
        )

        # 4. 创建 Bucket
        #
        # 注意：
        # 当前 oss2 SDK 的第二个参数叫 input
        # 不能写 bucket_config=
        #
        bucket.create_bucket(
            acl,
            bucket_config
        )

        print()
        print("=" * 60)
        print("Bucket 创建成功")
        print("=" * 60)
        print(f"Bucket       : {bucket_name}")
        print(f"Endpoint     : {endpoint}")
        print(f"Storage Class: {storage_class}")
        print(f"ACL          : {acl}")
        print("=" * 60)

    except oss2.exceptions.ServerError as e:

        print()
        print("=" * 60)
        print("OSS 服务器返回错误")
        print("=" * 60)

        print(f"HTTP Status : {e.status}")
        print(f"Error Code  : {e.code}")
        print(f"Message     : {e.message}")
        print(f"Request ID  : {e.request_id}")

    except oss2.exceptions.ClientError as e:

        print()
        print("OSS 客户端错误：")
        print(e)

    except Exception as e:

        print()
        print("发生未预期的错误：")
        print(f"Type: {type(e).__name__}")
        print(f"Error: {e}")


if __name__ == "__main__":
    create_oss_bucket(
        bucket_name=BUCKET_NAME,
        endpoint=ENDPOINT,
        access_key_id=ACCESS_KEY_ID,
        access_key_secret=ACCESS_KEY_SECRET,
        storage_class=oss2.BUCKET_STORAGE_CLASS_STANDARD,
        acl=oss2.BUCKET_ACL_PRIVATE,
    )
