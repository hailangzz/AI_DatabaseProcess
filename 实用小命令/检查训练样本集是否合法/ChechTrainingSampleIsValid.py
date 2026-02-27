import os
from PIL import Image
from tqdm import tqdm


def check_dataset(images_dir, labels_dir, mode="detect", log_file=None):
    """
    检查数据集质量，支持 YOLOv8 检测和语义分割
    images_dir: 图像目录
    labels_dir: 标签目录
    mode: "detect" 或 "segment"
    log_file: 可选，保存异常信息
    """
    invalid_images = []
    invalid_labels = []

    img_list = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    for img_name in tqdm(img_list, desc="Checking dataset"):
        img_path = os.path.join(images_dir, img_name)
        label_base = os.path.splitext(img_name)[0]

        # 1️⃣ 检查图像
        try:
            with Image.open(img_path) as im:
                width, height = im.size
                if width <= 0 or height <= 0:
                    raise ValueError("Image has zero width or height")
        except Exception as e:
            invalid_images.append((img_name, str(e)))
            continue  # 图像打不开就跳过标签检查

        # 2️⃣ 检查标签
        if mode == "detect":
            label_path = os.path.join(labels_dir, label_base + ".txt")
            if not os.path.exists(label_path):
                invalid_labels.append((img_name, "Label file not found"))
                continue
            try:
                with open(label_path, "r") as f:
                    lines = [line.strip() for line in f if line.strip() != ""]
                for i, line in enumerate(lines):
                    parts = line.split()
                    if len(parts) != 5:
                        raise ValueError(f"Line {i + 1} does not have 5 elements")
                    cls, x_center, y_center, w, h = parts
                    x_center, y_center, w, h = map(float, [x_center, y_center, w, h])
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        raise ValueError(f"Line {i + 1} coordinates out of [0,1] range")
                    if w <= 0 or h <= 0:
                        raise ValueError(f"Line {i + 1} width/height <= 0")
            except Exception as e:
                invalid_labels.append((img_name, str(e)))

        elif mode == "segment":
            # 支持 mask png 或 npy 文件
            mask_path_png = os.path.join(labels_dir, label_base + ".png")
            mask_path_npy = os.path.join(labels_dir, label_base + ".npy")
            mask_path = None
            if os.path.exists(mask_path_png):
                mask_path = mask_path_png
            elif os.path.exists(mask_path_npy):
                mask_path = mask_path_npy
            else:
                invalid_labels.append((img_name, "Mask file not found"))
                continue
            try:
                if mask_path.endswith(".png"):
                    with Image.open(mask_path) as m:
                        m_width, m_height = m.size
                        if m_width != width or m_height != height:
                            raise ValueError(f"Mask size {m_width}x{m_height} != image size {width}x{height}")
                elif mask_path.endswith(".npy"):
                    import numpy as np
                    mask = np.load(mask_path)
                    if mask.shape[:2] != (height, width):
                        raise ValueError(f"Mask shape {mask.shape[:2]} != image size {height}x{width}")
            except Exception as e:
                invalid_labels.append((img_name, str(e)))

        else:
            raise ValueError(f"Unknown mode: {mode}, must be 'detect' or 'segment'")

    # 输出结果
    print(f"\nInvalid images: {len(invalid_images)}")
    for img, reason in invalid_images:
        print(f"  {img}: {reason}")

    print(f"Invalid labels: {len(invalid_labels)}")
    for img, reason in invalid_labels:
        print(f"  {img}: {reason}")

    # 保存日志
    if log_file:
        with open(log_file, "w") as f:
            f.write("Invalid images:\n")
            for img, reason in invalid_images:
                f.write(f"{img}: {reason}\n")
            f.write("\nInvalid labels:\n")
            for img, reason in invalid_labels:
                f.write(f"{img}: {reason}\n")

    print("Dataset check completed.")
    return invalid_images, invalid_labels


# 使用示例
images_dir = "/home/chenkejing/database/AITotal_SegmentDatabase/carpetDatabaseSegment/images/train"
labels_dir = "/home/chenkejing/database/AITotal_SegmentDatabase/carpetDatabaseSegment/labels/train"
# mode="detect" 或 mode="segment"
invalid_imgs, invalid_labs = check_dataset(images_dir, labels_dir, mode="segment", log_file="dataset_check.log")
