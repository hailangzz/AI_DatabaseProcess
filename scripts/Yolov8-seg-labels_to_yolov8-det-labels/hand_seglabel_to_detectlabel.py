import os
import glob


def seg_to_det(seg_label_path, save_dir):
    """
    将 YOLOv8-seg 标注转换为 YOLOv8 detection 标注

    :param seg_label_path: segmentation标注文件夹路径
    :param save_dir: 保存 detection 标注文件夹路径
    """

    os.makedirs(save_dir, exist_ok=True)

    label_files = glob.glob(os.path.join(seg_label_path, "*.txt"))

    for label_file in label_files:
        with open(label_file, "r") as f:
            lines = f.readlines()

        new_lines = []

        for line in lines:
            parts = line.strip().split()

            if len(parts) < 7:
                # 至少要一个三角形（3个点 = 6个数 + class）
                continue

            cls_id = parts[0]
            coords = list(map(float, parts[1:]))

            xs = coords[0::2]
            ys = coords[1::2]

            x_min = min(xs)
            x_max = max(xs)
            y_min = min(ys)
            y_max = max(ys)

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            new_line = f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            new_lines.append(new_line)

        save_path = os.path.join(save_dir, os.path.basename(label_file))

        with open(save_path, "w") as f:
            f.write("\n".join(new_lines))

    print("转换完成！")

seg_label_dir = "/home/chenkejing/database/Negativew_Example_Dataset/hand/segment_Negative_hand_database/labels"
save_det_dir = "/home/chenkejing/database/Negativew_Example_Dataset/hand/segment_Negative_hand_database/save_detection_labels"

seg_to_det(seg_label_dir, save_det_dir)
