# -*- coding: utf-8 -*-

import shutil
from pathlib import Path

# =========================
# 配置
# =========================

SOURCE_IMAGES_DIR = Path(
    "/home/chenkejing/Downloads/CrowdHuman/CrowdHuman_train03/Images"
)

TARGET_IMAGES_DIR = Path("/home/chenkejing/Downloads/CrowdHuman/images")

SOURCE_LABELS_DIR = Path(
    "/home/chenkejing/Downloads/CrowdHuman/CrowdHuman_train03/Images"
)

TARGET_LABELS_DIR = Path("/home/chenkejing/Downloads/CrowdHuman/labels")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

LABEL_EXTS = {".txt"}


def copy_files(source_dir: Path, target_dir: Path, extensions: set, prefix=""):
    """
    复制指定类型文件

    """

    target_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for src_file in source_dir.rglob("*"):

        if not src_file.is_file():
            continue

        if src_file.suffix.lower() not in extensions:
            continue

        dst_file = target_dir / src_file.name

        # 已存在
        if dst_file.exists():
            skipped += 1

            print(f"[skip] {dst_file}")

            continue

        shutil.copy2(src_file, dst_file)

        copied += 1

        print(f"[{prefix}] {src_file.name}")

    print("\n====================")
    print(f"{prefix} 完成")
    print(f"复制: {copied}")
    print(f"跳过: {skipped}")
    print("====================\n")


if __name__ == "__main__":
    # copy images

    copy_files(SOURCE_IMAGES_DIR, TARGET_IMAGES_DIR, IMAGE_EXTS, prefix="IMAGE")

    # copy labels

    copy_files(SOURCE_LABELS_DIR, TARGET_LABELS_DIR, LABEL_EXTS, prefix="LABEL")

    print("✅ 全部完成")
