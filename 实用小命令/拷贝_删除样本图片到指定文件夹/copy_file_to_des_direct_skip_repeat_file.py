import os
import shutil

# ====== 配置路径 ======
source_images_dir = r"/home/chenkejing/database/AITotal_SegmentDatabase/wireDatabaseSegment_old/images/train"
target_images_dir = r"/data/database/AITotal_SegmentDatabase/wireDatabaseSegment/images/train"

# ====== 配置路径 ======
source_labels_dir = r"/home/chenkejing/database/AITotal_SegmentDatabase/wireDatabaseSegment_old/labels/train"
target_labels_dir = r"/data/database/AITotal_SegmentDatabase/wireDatabaseSegment/labels/train"

# 支持的图片格式
image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".txt"}


def copy_images(src, dst):
    copied_count = 0
    skipped_count = 0

    if not os.path.exists(dst):
        os.makedirs(dst)

    for root, dirs, files in os.walk(src):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_exts:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(dst, file)

                # ✅ 如果目标中已存在同名文件，则跳过
                if os.path.exists(dst_path):
                    skipped_count += 1
                    print(f"跳过（已存在）: {dst_path}")
                    continue

                # 执行复制
                shutil.copy2(src_path, dst_path)
                copied_count += 1
                print(f"已复制: {src_path} -> {dst_path}")

    print(f"\n拷贝完成！")
    print(f"成功复制数量: {copied_count}")
    print(f"跳过数量: {skipped_count}")


if __name__ == "__main__":
    # copy_images(source_images_dir, target_images_dir)
    copy_images(source_labels_dir, target_labels_dir)
    print("✅ 图片复制完成！")