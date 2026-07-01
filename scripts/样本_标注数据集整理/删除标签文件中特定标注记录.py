"""
删除YOLO标注文件中：
1. 指定类别的标注
2. 包含指定数量空格的标注行
"""

from pathlib import Path


def process_yolo_labels(label_dir):
    label_dir = Path(label_dir)

    txt_files = list(label_dir.glob("*.txt"))

    print(f"发现 {len(txt_files)} 个标注文件")

    total_removed = 0

    for txt_path in txt_files:

        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        removed_count = 0

        for line in lines:

            line = line.strip()

            # 跳过空行
            if not line:
                continue

            remove_flag = False

            # ------------------------------------------------
            # 按空格数量删除
            # ------------------------------------------------
            if ENABLE_REMOVE_BY_SPACE_COUNT:
                space_count = line.count(" ")

                if space_count == REMOVE_SPACE_COUNT:
                    remove_flag = True

            # ------------------------------------------------
            # 按类别删除
            # ------------------------------------------------
            if ENABLE_REMOVE_CLASS:

                try:
                    cls_id = int(line.split()[0])

                    if cls_id in REMOVE_CLASSES:
                        remove_flag = True

                except Exception:
                    print(f"[WARNING] 无法解析类别：{txt_path.name} -> {line}")

            # ------------------------------------------------
            # 删除或保留
            # ------------------------------------------------
            if remove_flag:
                removed_count += 1
                print(f"[REMOVE] {txt_path.name} -> {line}")
            else:
                new_lines.append(line + "\n")

        total_removed += removed_count

        # 保存修改后的文件
        if SAVE_CHANGES:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

    print("\n==============================")
    print(f"总共删除 {total_removed} 条标注")
    print("处理完成")
    print("==============================")


if __name__ == "__main__":
    # ==========================================
    # 配置区
    # ==========================================

    LABEL_DIR = "/data/database/PlasticBagDatabase/plastic bag.v3i.coco-segmentation/train/yolov8_labels/seg"

    # True：真正修改文件
    # False：仅打印，不修改
    SAVE_CHANGES = True

    # ------------------------------------------
    # 按类别删除
    # ------------------------------------------
    ENABLE_REMOVE_CLASS = False

    # 要删除的类别
    REMOVE_CLASSES = {3, 5}

    # ------------------------------------------
    # 按空格数量删除
    # ------------------------------------------
    ENABLE_REMOVE_BY_SPACE_COUNT = True

    # 删除包含多少个空格的行
    # 例如：
    # 10 -> 删除包含10个空格的行
    # 20 -> 删除包含20个空格的行
    REMOVE_SPACE_COUNT = 8

    # ==========================================
    # 开始处理
    # ==========================================

    process_yolo_labels(LABEL_DIR)
