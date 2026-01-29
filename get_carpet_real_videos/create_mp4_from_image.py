import cv2
import os

# ---------- 配置 ----------
image_folder = '/home/chenkejing/PycharmProjects/ultralytics/results/carpet'  # 图像所在文件夹
# output_video = 'output_video.mp4'  # 输出的 MP4 视频文件名
output_video = os.path.join(image_folder,'carpet_output_video.mp4')
frame_rate = 5  # 帧率，单位为帧/秒

# 获取图像文件列表并按名称排序
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # 根据文件名中的数字部分排序

# 如果没有找到图像文件，打印提示并退出
if not image_files:
    print("No images found in the folder.")
    exit()

# 读取第一张图片，获取图像的尺寸（宽度和高度）
first_image_path = os.path.join(image_folder, image_files[0])
first_image = cv2.imread(first_image_path)
height, width, _ = first_image.shape

# 创建视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码格式
video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

# 读取并写入每一张图像
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)

    # 确保图像的尺寸和第一张图片一致
    image = cv2.resize(image, (width, height))

    # 写入视频
    video_writer.write(image)
    print(f"Adding image {image_file} to video.")

# 释放视频写入器
video_writer.release()
print(f"Video saved as {output_video}")
