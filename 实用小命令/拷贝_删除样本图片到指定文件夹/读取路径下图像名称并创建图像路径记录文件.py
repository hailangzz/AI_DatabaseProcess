# -*- coding: utf-8 -*-

"""
功能：
1. 扫描指定目录下的所有图片
2. 将图片完整路径写入 total_images_path_info.txt
3. 增量写入，已经存在的图片路径不会重复写入
4. 支持递归扫描子目录
"""

from pathlib import Path

# 支持的图片格式
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# ============================================================
# 读取已经存在的图片路径
# ============================================================

def load_existing_paths(output_file):
    """
    读取 total_images_path_info.txt 中已经存在的路径。
    """

    existing_paths = set()

    if not output_file.exists():
        return existing_paths

    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            path = line.strip()

            if path:
                existing_paths.add(path)

    return existing_paths


# ============================================================
# 扫描图片
# ============================================================

def find_all_images(root_dir):
    """
    递归扫描 root_dir 下所有图片。
    """

    image_paths = []

    for path in root_dir.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # 不把输出文件当成图片，实际上它不是图片，这里只是保险
        if path.name == "total_images_path_info.txt":
            continue

        image_paths.append(path)

    return image_paths


# ============================================================
# 主逻辑
# ============================================================

def main():
    if not IMAGE_ROOT_DIR.exists():
        print(f"[ERROR] 图片目录不存在：{IMAGE_ROOT_DIR}")
        return

    print(f"[INFO] 扫描目录：{IMAGE_ROOT_DIR}")

    # 读取历史记录
    existing_paths = load_existing_paths(OUTPUT_FILE)

    print(f"[INFO] 已存在路径数量：{len(existing_paths)}")

    # 扫描所有图片
    image_paths = find_all_images(IMAGE_ROOT_DIR)

    print(f"[INFO] 当前扫描到图片数量：{len(image_paths)}")

    # 找出新增图片
    new_paths = []

    for image_path in image_paths:

        # 使用绝对路径
        absolute_path = str(image_path.resolve())

        if absolute_path in existing_paths:
            continue

        new_paths.append(absolute_path)

    # 没有新增图片
    if not new_paths:
        print("[INFO] 没有发现新的图片路径，无需写入。")
        return

    # 增量追加
    with OUTPUT_FILE.open("a", encoding="utf-8") as f:

        for image_path in new_paths:
            f.write(image_path + "\n")

    print(f"[INFO] 新增图片路径：{len(new_paths)}")
    print(f"[INFO] 已写入：{OUTPUT_FILE}")


# ============================================================
# 配置
# ============================================================

# 要扫描的图片目录
IMAGE_ROOT_DIR = Path(
    "/data/database/aws_origin_sample/images"
)

# 输出文件
OUTPUT_FILE = IMAGE_ROOT_DIR / "total_images_path_info.txt"

if __name__ == "__main__":
    main()
