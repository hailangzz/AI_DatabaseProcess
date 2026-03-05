import os

# ====== 配置路径 ======
source_dir = "/home/chenkejing/database/WireDatabase/Wildlife Monitoring and Poaching Detection.v8-final-version/train/imgs"   # 目录A：读取图片名称
delete_images_target_dir = "/home/chenkejing/database/WireDatabase/TotalPublicWireDatabase/images"   # 目录B：删除同名图片

# 支持的图片格式
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def get_image_names(directory):
    """获取目录下所有图片文件名（不含路径）"""
    image_names = set()
    for file in os.listdir(directory):
        ext = os.path.splitext(file)[1].lower()
        if ext in IMG_EXTENSIONS:
            image_names.add(file)
    return image_names


def delete_same_images(source_dir, target_dir, dry_run=False):
    """
    删除 target_dir 中与 source_dir 同名的图片
    dry_run=True 时只预览，不执行删除
    """
    source_images = get_image_names(source_dir)

    deleted_count = 0
    checked_count = 0

    for file in os.listdir(target_dir):
        ext = os.path.splitext(file)[1].lower()
        if ext not in IMG_EXTENSIONS:
            continue

        checked_count += 1

        if file in source_images:
            file_path = os.path.join(target_dir, file)

            if dry_run:
                print(f"[预览] 将删除: {file_path}")
            else:
                os.remove(file_path)
                print(f"[删除] {file_path}")

            deleted_count += 1

    print("\n===== 统计 =====")
    print(f"检查图片数量: {checked_count}")
    print(f"删除图片数量: {deleted_count}")


if __name__ == "__main__":
    # 👉 先用 dry_run=True 预览
    delete_same_images(source_dir, delete_images_target_dir, dry_run=False)

    # 确认无误后再执行删除
    # delete_same_images(source_dir, target_dir, dry_run=False)