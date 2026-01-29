"""
说明：批量将图片 resize 到指定尺寸并保存到目标目录

功能：
1. 遍历指定图片文件夹
2. 将图片 resize 为指定宽高
3. 保存到指定目录，保持原文件名
"""

import os
import cv2
import argparse

# 默认路径（可修改为你的实际路径）
DEFAULT_IMG_DIR = "./images"
DEFAULT_SAVE_DIR = "./resized_images"

DEFAULT_IMG_WIDTH = 640
DEFAULT_IMG_HEIGHT = 480


def resize_images(img_dir, save_dir, width, height):
    """
    img_dir: 图片所在目录
    save_dir: 保存目录
    width, height: 目标尺寸
    """
    # 支持的图片格式
    exts = ('.jpg', '.jpeg', '.png', '.bmp')

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 遍历图片
    img_list = [f for f in os.listdir(img_dir) if f.lower().endswith(exts)]
    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[Warning] Failed to read {img_path}")
            continue

        # resize 图片
        resized = cv2.resize(img, (width, height))

        # 保存
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, resized)

    print(f"Resized {len(img_list)} images to {width}x{height} and saved in '{save_dir}'")


def parse_args():
    parser = argparse.ArgumentParser(description="Resize images to specified size")
    parser.add_argument("--img_dir", type=str, default=DEFAULT_IMG_DIR, help=f"Directory containing images (default: {DEFAULT_IMG_DIR})")
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR, help=f"Directory to save resized images (default: {DEFAULT_SAVE_DIR})")
    parser.add_argument("--width", type=int, default= DEFAULT_IMG_WIDTH,help="Target width")
    parser.add_argument("--height", type=int, default= DEFAULT_IMG_HEIGHT,help="Target height")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resize_images(args.img_dir, args.save_dir, args.width, args.height)
