import os
import argparse

# 用法：python count_images.py --img_dir /path/to/images
# 默认路径（可修改为你常用的图片目录）
DEFAULT_IMG_DIR = "/home/chenkejing/database/Negativew_Example_Dataset/carpet/Negative_carpet_database/images"


def count_images(img_dir):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    count = 0

    for name in os.listdir(img_dir):
        if name.lower().endswith(exts):
            count += 1

    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count images in a directory")
    parser.add_argument(
        "--img_dir",
        type=str,
        default=DEFAULT_IMG_DIR,  # 设置默认路径
        help=f"Image directory (default: {DEFAULT_IMG_DIR})"
    )
    args = parser.parse_args()

    img_dir = args.img_dir
    if not os.path.exists(img_dir):
        print(f"Error: directory '{img_dir}' does not exist!")
    else:
        num = count_images(img_dir)
        print(f"Image count in '{img_dir}': {num}")
