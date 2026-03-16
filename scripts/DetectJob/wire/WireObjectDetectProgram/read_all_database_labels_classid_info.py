import os
import utils.util as util

#将项目使用到的数据集，标注信息、样本数量、照片数量等信息，记录到class_counts.json中
# ===== 示例调用 =====
if __name__ == "__main__":
    dataset_name = "Wildlife Monitoring and Poaching Detection.v8-final-version"  # 数据集名称
    label_dir = "/home/chenkejing/database/Wildlife Monitoring and Poaching Detection.v8-final-version/train/labels"# YOLO txt 文件夹
    output_json = "class_counts.json"

    counts = util.count_yolo_class_ids(label_dir)
    util.save_dataset_class_counts_json(dataset_name, counts, output_json)