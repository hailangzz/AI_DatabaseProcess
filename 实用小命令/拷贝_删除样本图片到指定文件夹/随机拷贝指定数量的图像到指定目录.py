# 说明：随机拷贝指定数量的图像到指定目录。
# 作用：在生成量化数据集时，一般用1000张样本。随机从真实数据集中提取样本生成量化数据集

import os
import shutil
import random
import argparse

def random_copy_images(src_dir, dst_dir, num_images):
    # 支持的图片格式
    exts = ('.jpg', '.jpeg', '.png', '.bmp')

    # 获取所有图片
    all_images = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith(exts)
    ]

    total = len(all_images)
    print(f"[INFO] Found {total} images")

    if total == 0:
        print("[ERROR] No images found!")
        return

    # 如果数量超过，自动取最大
    if num_images > total:
        print(f"[WARN] Requested {num_images}, but only {total} available. Using all.")
        num_images = total

    # 随机选择
    selected = random.sample(all_images, num_images)

    # 创建目标目录
    os.makedirs(dst_dir, exist_ok=True)

    # 拷贝
    for i, img_name in enumerate(selected):
        src_path = os.path.join(src_dir, img_name)
        dst_path = os.path.join(dst_dir, img_name)

        shutil.copy2(src_path, dst_path)

        print(f"[{i+1}/{num_images}] Copied: {img_name}")

    print(f"[DONE] Copied {num_images} images to {dst_dir}")


default_src = "/home/chenkejing/database/HandDetect/EmdoorRealHandImages/unshare_images/train/images"
default_dst = "/home/chenkejing/PycharmProjects/ultralytics/images_mode_test/hand_real_image"
default_num = 250

def parse_args():
    parser = argparse.ArgumentParser(description="Randomly copy images")

    parser.add_argument("--src", type=str, default=default_src, help="Source directory")
    parser.add_argument("--dst", type=str, default=default_dst, help="Destination directory")
    parser.add_argument("--num", type=int, default=default_num, help="Number of images to copy")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    random_copy_images(args.src, args.dst, args.num)

