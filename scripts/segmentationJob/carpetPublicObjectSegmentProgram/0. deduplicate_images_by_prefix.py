import os
from collections import defaultdict
from pathlib import Path

# ==============================
# 配置区
# ==============================
IMAGE_DIR = "/home/chenkejing/database/carpetDatabase/PublicCarpetDatabase_Myself/images"   # 修改为你的图像目录
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DRY_RUN = False  # True = 只打印不删除；False = 真正删除

# ==============================
# 主逻辑
# ==============================
def deduplicate_images(image_dir: str):
    image_dir = Path(image_dir)
    groups = defaultdict(list)

    # 1. 收集图片并分组
    for file in image_dir.iterdir():
        if not file.is_file():
            continue
        if file.suffix.lower() not in IMAGE_EXTS:
            continue

        prefix = file.name.split(".", 1)[0]  # 第一个 . 之前
        size = file.stat().st_size
        groups[prefix].append((file, size))

    # 2. 处理每一组
    removed_files = []
    kept_files = []

    for prefix, files in groups.items():
        if len(files) <= 1:
            kept_files.append(files[0][0])
            continue

        # 按文件大小排序（降序）
        files.sort(key=lambda x: x[1], reverse=True)

        keep_file = files[0][0]
        kept_files.append(keep_file)

        for f, _ in files[1:]:
            removed_files.append(f)

    # 3. 输出 & 删除
    print(f"[INFO] 总分组数: {len(groups)}")
    print(f"[INFO] 保留图片数: {len(kept_files)}")
    print(f"[INFO] 待删除图片数: {len(removed_files)}")

    for f in removed_files:
        print(f"[REMOVE] {f.name}")
        if not DRY_RUN:
            f.unlink()

    print("[DONE] 去重完成")

# ==============================
# 入口
# ==============================
if __name__ == "__main__":
    deduplicate_images(IMAGE_DIR)
