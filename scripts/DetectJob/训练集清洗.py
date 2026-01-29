# 说明：此脚本用于，在模型训练前，对整个训练数据集，标签、标注文件、等越界问题，进行清洗整理，防止因标注文件问题。导致GPU训练崩溃。

import os

label_dir = "/home/chenkejing/database/AITotal_ProjectDatabase/carpetDatabaseProgrem/labels/train"

for file in os.listdir(label_dir):
    if not file.endswith(".txt"):
        continue

    path = os.path.join(label_dir, file)
    new_lines = []

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, x, y, w, h = map(float, parts)

            # class 修正
            cls = 0

            # bbox 合法性
            if w <= 0 or h <= 0:
                continue
            if not (0 <= x <= 1 and 0 <= y <= 1 and w <= 1 and h <= 1):
                continue

            new_lines.append(f"{int(cls)} {x} {y} {w} {h}")

    with open(path, "w") as f:
        f.write("\n".join(new_lines))
