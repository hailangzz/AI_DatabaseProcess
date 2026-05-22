# 说明：此代码将模型在现场实测、运行时的基于位置进行预测存储的图像进行分类整理。用于计算实测条件下的误检率、漏检率等模型性能指标；

import os
import shutil

# =========================
# 配置区域
# =========================

# 文件夹路径
folder_path = "/home/chenkejing/Downloads/WireSegmentProject/spatial_location_val_images"

# 两个特定字符串
keyword1 = "exist_target"
keyword2 = "null_target"

# 支持的图片后缀
image_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".txt"   # 增加上标注文件，移动处理
)

# =========================
# 创建目标文件夹
# =========================

target_dir1 = os.path.join(folder_path, keyword1)
target_dir2 = os.path.join(folder_path, keyword2)

os.makedirs(target_dir1, exist_ok=True)
os.makedirs(target_dir2, exist_ok=True)

# =========================
# 遍历文件
# =========================

for filename in os.listdir(folder_path):

    file_path = os.path.join(folder_path, filename)

    # 跳过文件夹
    if not os.path.isfile(file_path):
        continue

    # 判断是否为图片
    if not filename.lower().endswith(image_extensions):
        continue

    # =========================
    # 移动包含 keyword1 的图片
    # =========================
    if keyword1 in filename:

        dst_path = os.path.join(target_dir1, filename)

        shutil.move(file_path, dst_path)

        print(f"Moved to {keyword1}: {filename}")

    # =========================
    # 移动包含 keyword2 的图片
    # =========================
    elif keyword2 in filename:

        dst_path = os.path.join(target_dir2, filename)

        shutil.move(file_path, dst_path)

        print(f"Moved to {keyword2}: {filename}")

print("Done.")