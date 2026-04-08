"""
空样本占比建议控制在：10% ~ 30%

✅ 不同占比的效果（很实用）
空样本占比	模型表现	适用场景
0%	❌ 误检严重	不推荐
5% 以下	⚠️ 仍容易误检	轻微改善
10% ~ 20%	✅ 最均衡	⭐ 推荐
20% ~ 30%	✅ 强抑制误检	背景复杂场景
> 40%	❌ 漏检增加	模型变保守
"""

import os
import shutil
from tqdm import tqdm

def filter_yolov8_seg_dataset(
    img_dir,
    label_dir,
    output_dir,
    target_classes=None,
    reindex=True,
    keep_empty=False,
    only_empty=False  # ⭐ 新增：只保留“原始空label”
):
    """
    YOLOv8-seg 数据筛选工具（增强版）

    参数：
        img_dir: 图片路径
        label_dir: 标签路径
        output_dir: 输出路径
        target_classes: 需要保留的类别，如 [0,1]
        reindex: 是否重新编号类别
        keep_empty: 是否保留空样本
        only_empty: ⭐ 是否只保留“原始空label样本”
    """

    out_img_dir = os.path.join(output_dir, "images")
    out_label_dir = os.path.join(output_dir, "labels")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    if target_classes is None:
        target_classes = []

    # 类别映射
    if reindex and len(target_classes) > 0:
        class_map = {cls: i for i, cls in enumerate(target_classes)}
    else:
        class_map = {cls: cls for cls in target_classes}

    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

    kept_images = 0
    empty_images = 0

    for label_file in tqdm(label_files):
        label_path = os.path.join(label_dir, label_file)

        with open(label_path, "r") as f:
            lines = f.readlines()

        base_name = os.path.splitext(label_file)[0]

        # 找对应图片
        img_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            temp_path = os.path.join(img_dir, base_name + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                break

        if img_path is None:
            print(f"⚠️ Image not found for {label_file}")
            continue

        # ⭐ 是否原始空label
        is_original_empty = (len(lines) == 0)

        # ==================================================
        # ⭐ 模式1：只提取“真正空样本”
        # ==================================================
        if only_empty:
            if is_original_empty:
                out_label_path = os.path.join(out_label_dir, label_file)
                open(out_label_path, "w").close()

                shutil.copy(img_path, os.path.join(out_img_dir, os.path.basename(img_path)))
                empty_images += 1
            continue

        # ==================================================
        # ⭐ 模式2：正常筛选类别
        # ==================================================
        new_lines = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) == 0:
                continue

            cls_id = int(parts[0])

            if cls_id in target_classes:
                parts[0] = str(class_map.get(cls_id, cls_id))
                new_lines.append(" ".join(parts))

        # ⭐ 有目标类别
        if len(new_lines) > 0:
            out_label_path = os.path.join(out_label_dir, label_file)
            with open(out_label_path, "w") as f:
                f.write("\n".join(new_lines))

            shutil.copy(img_path, os.path.join(out_img_dir, os.path.basename(img_path)))
            kept_images += 1

        # ⭐ 空样本（但不是原始空）
        elif keep_empty:
            out_label_path = os.path.join(out_label_dir, label_file)
            open(out_label_path, "w").close()

            shutil.copy(img_path, os.path.join(out_img_dir, os.path.basename(img_path)))
            empty_images += 1

    print("\n✅ Done!")
    print(f"🎯 Positive images: {kept_images}")
    print(f"🟦 Empty images: {empty_images}")


## 当只提取空样本时
# 1    ️⃣ ⭐ 只提取“真正空白样本”（你当前需求）
filter_yolov8_seg_dataset(
    img_dir="/home/chenkejing/database/No_Target_Example_Dataset/No_Target_database/NO_target_camera_images_0407_batch1/images",
    label_dir="/home/chenkejing/database/No_Target_Example_Dataset/No_Target_database/NO_target_camera_images_0407_batch1/yolov8_labels/seg",
    output_dir="/home/chenkejing/database/No_Target_Example_Dataset/No_Target_database/NO_target_camera_images_0407_batch1_output_empty_only",
    only_empty=True
)

"""
    # 👉 结果：只包含原始label，文件为空的图片，不会混入其他类别 ❗（推荐）   提取指定类别（不保留空样本）
    filter_yolov8_seg_dataset(
        img_dir="images",
        label_dir="labels",
        output_dir="output_class0",
        target_classes=[0],
        keep_empty=False
    )

    ## 3 ️⃣ 提取类别 + 保留空样本（常用）
    filter_yolov8_seg_dataset(
        img_dir="images",
        label_dir="labels",
        output_dir="output_mix",
        target_classes=[0],
        keep_empty=True
    )
"""