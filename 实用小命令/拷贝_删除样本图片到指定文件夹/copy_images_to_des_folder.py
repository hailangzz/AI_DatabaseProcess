import os
import shutil

# ====== 配置路径 ======
source_dir = r"/home/chenkejing/database/WireDatabase/Wildlife Monitoring and Poaching Detection.v8-final-version/train"      # 原始图片目录
target_dir = r"/home/chenkejing/database/WireDatabase/TotalPublicWireDatabase/images"      # 目标目录

# 支持的图片格式
image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def copy_images(src, dst):
    image_count = 0
    if not os.path.exists(dst):
        os.makedirs(dst)

    for root, dirs, files in os.walk(src):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_exts:
                image_count+=1
                src_path = os.path.join(root, file)

                # 处理重名文件
                base_name = os.path.splitext(file)[0]
                new_name = file
                count = 1
                while os.path.exists(os.path.join(dst, new_name)):
                    new_name = f"{base_name}_{count}{ext}"
                    count += 1

                dst_path = os.path.join(dst, new_name)

                shutil.copy2(src_path, dst_path)
                print(f"已复制: {src_path} -> {dst_path}")
    print("拷贝的图像数量：%d",image_count)


if __name__ == "__main__":
    copy_images(source_dir, target_dir)
    print("✅ 图片复制完成！")