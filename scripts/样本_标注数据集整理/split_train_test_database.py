import os
import random
import shutil

import os
import random
import shutil

def split_yolo_dataset(
    image_dir,
    label_dir,
    output_dir,
    test_ratio=0.2,
    seed=42
):
    """
    将 YOLO 数据集随机划分出测试集（使用剪切）

    参数：
        image_dir: 图片目录 (images)
        label_dir: 标签目录 (labels)
        output_dir: 输出目录
        test_ratio: 测试集比例 (0~1)
        seed: 随机种子
    """

    random.seed(seed)

    # 输出路径
    test_image_dir = os.path.join(output_dir, "images/val")
    test_label_dir = os.path.join(output_dir, "labels/val")

    os.makedirs(test_image_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)

    # 支持的图片格式
    image_exts = [".jpg", ".jpeg", ".png", ".bmp"]

    # 获取所有图片文件
    images = [
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in image_exts
    ]

    print(f"总图片数量: {len(images)}")

    # 随机打乱
    random.shuffle(images)

    # 计算测试集数量
    test_num = int(len(images) * test_ratio)
    test_images = images[:test_num]

    print(f"测试集数量: {test_num}")

    # 开始剪切
    missing_labels = 0

    for img_name in test_images:
        img_path = os.path.join(image_dir, img_name)

        # 对应 label 文件
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)

        # 剪切图片
        shutil.move(img_path, os.path.join(test_image_dir, img_name))

        # 剪切标签（如果存在）
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(test_label_dir, label_name))
        else:
            missing_labels += 1

    print(f"完成！缺失标签数量: {missing_labels}")
    print(f"测试集路径: {output_dir}/val")


if __name__ == "__main__":
    split_yolo_dataset(
        image_dir="/home/chenkejing/database/AITotal_SegmentDatabase/wireDatabaseSegment/images/train",
        label_dir="/home/chenkejing/database/AITotal_SegmentDatabase/wireDatabaseSegment/labels/train",
        output_dir="/home/chenkejing/database/AITotal_SegmentDatabase/wireDatabaseSegment",
        test_ratio=0.05  # 20%作为测试集
    )