import os


def rename_files(directory, old_str, new_str):
    """
    将指定目录下所有图片和 txt 文件名称中的 old_str
    替换为 new_str。

    参数：
        directory: 目录路径
        old_str:   要被替换的字符串
        new_str:   替换后的字符串
    """

    # 支持的图片格式
    image_exts = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    # 支持的文件格式
    valid_exts = image_exts | {".txt"}

    if not os.path.isdir(directory):
        print(f"目录不存在：{directory}")
        return

    for filename in os.listdir(directory):

        file_path = os.path.join(directory, filename)

        # 跳过目录
        if not os.path.isfile(file_path):
            continue

        # 获取文件扩展名
        ext = os.path.splitext(filename)[1].lower()

        # 只处理图片和 txt
        if ext not in valid_exts:
            continue

        # 文件名中没有目标字符串
        if old_str not in filename:
            continue

        # 生成新的文件名
        new_filename = filename.replace(old_str, new_str)

        new_file_path = os.path.join(
            directory,
            new_filename
        )

        # 防止覆盖已有文件
        if os.path.exists(new_file_path):
            print(f"跳过，目标文件已存在：{new_file_path}")
            continue

        os.rename(file_path, new_file_path)

        print(
            f"重命名：{filename} -> {new_filename}"
        )


rename_files(
    directory="/home/chenkejing/Downloads/WireSegmentProject/spatial_location_val_images/wire_detect/UT-A10XCNA00815T006/20260811/null",
    old_str="EMDOOR_COMMON",
    new_str="UT-A10XCNA00815T006"
)
