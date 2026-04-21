# 删除掉，没有标注文件的空图片（既没有目标类标注，也没有空标注文件）
# 保持图像、标注样本的统一
# 删除格式、存储错误的图像

import os
from pathlib import Path
from PIL import Image

# =====================
# 路径配置
# =====================
images_dir = Path("/data/database/AITotal_ProjectDatabase/finetune_random_sample_datebase/random_hand_database/images/train")
labels_dir = Path("/data/database/AITotal_ProjectDatabase/finetune_random_sample_datebase/random_hand_database/labels/train")

# 安全模式（True只打印，不删除）
# DRY_RUN = True
DRY_RUN = False  # 真正、删除文件

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =====================
# 工具函数
# =====================
def get_image_files(folder):
    return [f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMG_EXTS]

def get_label_stems(folder):
    return {f.stem for f in folder.glob("*.txt")}

def is_image_corrupt(img_path: Path):
    """
    尝试打开图片，判断是否损坏
    """
    try:
        with Image.open(img_path) as img:
            img.verify()  # 只验证，不加载像素
        return False
    except Exception:
        return True

def delete_file(path: Path):
    print(f"🗑 删除: {path}")
    if not DRY_RUN:
        try:
            path.unlink()
        except Exception as e:
            print(f"❌ 删除失败: {path} -> {e}")

# =====================
# 主逻辑
# =====================
def main():
    if not images_dir.exists() or not labels_dir.exists():
        print("❌ 目录不存在")
        return

    images = get_image_files(images_dir)
    label_stems = get_label_stems(labels_dir)

    image_stems = {img.stem for img in images}

    print(f"📊 图像数量: {len(images)}")
    print(f"📊 标注数量: {len(label_stems)}")

    # =====================
    # 1. 无标注图片
    # =====================
    img_without_label = image_stems - label_stems

    # =====================
    # 2. 无图片标注
    # =====================
    label_without_img = label_stems - image_stems

    # =====================
    # 3. 损坏图片检测
    # =====================
    corrupt_images = []
    for img in images:
        if is_image_corrupt(img):
            corrupt_images.append(img)

    print(f"🧹 无标注图片: {len(img_without_label)}")
    print(f"🧹 无图片标注: {len(label_without_img)}")
    print(f"💥 损坏图片: {len(corrupt_images)}")

    # =====================
    # 删除：无标注图片
    # =====================
    for stem in img_without_label:
        for ext in IMG_EXTS:
            img_path = images_dir / f"{stem}{ext}"
            if img_path.exists():
                delete_file(img_path)

    # =====================
    # 删除：无图片标注
    # =====================
    for stem in label_without_img:
        label_path = labels_dir / f"{stem}.txt"
        if label_path.exists():
            delete_file(label_path)

    # =====================
    # 删除：损坏图片 + 标注
    # =====================
    for img in corrupt_images:
        delete_file(img)

        label_path = labels_dir / f"{img.stem}.txt"
        if label_path.exists():
            delete_file(label_path)

    print("✅ 清理完成")

if __name__ == "__main__":
    main()