import os


def generate_image_path_txt(image_folder, base_path_str, output_txt_path):
    """
    读取指定文件夹内的所有图片名称，拼接基础路径字符串后写入 txt 文件。

    :param image_folder: 存放图片的文件夹路径
    :param base_path_str: 要拼接的前缀路径字符串
    :param output_txt_path: 导出的 txt 文件保存路径
    """
    # 常见的图像文件扩展名集合
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.gif')

    # 获取文件夹下所有指定格式的图像文件名
    image_names = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith(valid_extensions)
    ]

    # 按照文件名排序（可选，保持文件列表整洁）
    image_names.sort()

    # 写入 txt 文件
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for img_name in image_names:
            # 自动处理路径分隔符，将前缀与图片文件名拼接
            new_path = os.path.join(base_path_str, img_name)
            f.write(new_path + '\n')

    print(f"成功处理 {len(image_names)} 张图片，结果已保存至: {output_txt_path}")


# ================= 示例调用 =================
if __name__ == '__main__':
    # 1. 存放图片的本地文件夹路径
    img_dir = '/data/database/AITotal_Segment_ValDatabase/carpetValDatabaseSegment/images/val'

    # 2. 需要拼接的前缀路径字符串 (例如服务器/远程/相对路径字符串)
    base_prefix = '/data/database/AITotal_Segment_ValDatabase/carpetValDatabaseSegment/images/val'

    # 3. 生成的 txt 保存路径
    save_txt = '/data/database/AITotal_Segment_Hard_Case_Info/carpetDatabaseSegment/carpet_hard_cases.txt'

    # 执行函数
    generate_image_path_txt(img_dir, base_prefix, save_txt)
