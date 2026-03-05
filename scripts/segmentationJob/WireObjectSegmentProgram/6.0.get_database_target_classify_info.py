import os
from collections import defaultdict

def count_yolo_classes(label_dir):
    """
    统计YOLO格式数据集中各类别目标数量
    :param label_dir: labels目录路径
    """

    class_counts = defaultdict(int)
    total_images = 0
    total_objects = 0

    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue

        total_images += 1
        file_path = os.path.join(label_dir, file)

        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            class_id = int(parts[0])

            class_counts[class_id] += 1
            total_objects += 1

    # 打印统计结果
    print("========== YOLO 数据集统计 ==========")
    print(f"样本图片数量: {total_images}")
    print(f"目标总数量: {total_objects}")
    print("\n各类别目标数量:")

    for class_id in sorted(class_counts.keys()):
        print(f"类别 {class_id}: {class_counts[class_id]} 个")

    print("=====================================")


if __name__ == "__main__":
    label_dir = "/home/chenkejing/database/AITotal_SegmentDatabase/wireDatabaseSegment/labels/train"   # 修改为你的labels路径
    count_yolo_classes(label_dir)