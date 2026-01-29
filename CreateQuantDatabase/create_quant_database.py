import os

def save_image_paths(image_dir, output_txt):
    # 支持的图片格式
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    with open(output_txt, 'w', encoding='utf-8') as f:
        for filename in os.listdir(image_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in exts:
                relative_path = os.path.join(image_dir, filename)
                f.write(relative_path + "\n")

    print(f"Done! 写入了 {output_txt}")


if __name__ == "__main__":
    image_dir = "/home/chenkejing/PycharmProjects/ultralytics/images_mode_test/hand_real_image"      # ← 修改成你的图片目录
    output_txt = "/home/chenkejing/PycharmProjects/ultralytics/images_mode_test/hand_real_image/hand_quant.txt"   # ← 输出的txt文件名

    save_image_paths(image_dir, output_txt)
