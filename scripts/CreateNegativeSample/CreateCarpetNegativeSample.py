"""
说明：批量生成空的 YOLOv8 格式标注文件

功能：
1. 读取指定图片文件夹下的所有图片
2. 根据图片名生成对应的 .txt YOLOv8 标注文件
3. 标注文件内容为空
4. 保存到指定的标注文件夹

说明：python generate_empty_labels.py --img_dir /home/user/images --save_dir /home/user/labels
"""

import os
import argparse

# 默认路径（可修改为你的实际路径）
DEFAULT_IMG_DIR = "/home/chenkejing/database/Negativew_Example_Dataset/carpet/Negative_carpet_database/images"
DEFAULT_SAVE_DIR = "/home/chenkejing/database/Negativew_Example_Dataset/carpet/Negative_carpet_database/labels"


def generate_empty_yolo_labels(img_dir, save_dir):
    """
    img_dir: 图片所在目录
    save_dir: 标注文件保存目录
    """
    # 支持的图片格式
    exts = ('.jpg', '.jpeg', '.png', '.bmp')

    # 创建标注保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 遍历图片
    img_list = [f for f in os.listdir(img_dir) if f.lower().endswith(exts)]
    for img_name in img_list:
        # 去掉后缀，生成同名 txt 文件
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(save_dir, base_name + ".txt")

        # 写入空内容
        with open(label_path, 'w') as f:
            pass  # 空文件

    print(f"Generated empty YOLOv8 label files for {len(img_list)} images in '{save_dir}'")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate empty YOLOv8 label files for images")
    parser.add_argument(
        "--img_dir",
        type=str,
        default=DEFAULT_IMG_DIR,
        help=f"Directory containing images (default: {DEFAULT_IMG_DIR})"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=DEFAULT_SAVE_DIR,
        help=f"Directory to save label files (default: {DEFAULT_SAVE_DIR})"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_empty_yolo_labels(args.img_dir, args.save_dir)
