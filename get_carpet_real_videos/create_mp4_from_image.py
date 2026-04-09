import cv2
import os
import re


def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1


# ---------- 配置 ----------
image_folder = '/home/chenkejing/database/rosbag_info/main_carmera_images'

# 输出 AVI（MJPG 推荐用 avi 容器）
output_video = os.path.join(image_folder, 'MainCameraVideo.avi')

frame_rate = 5  # 帧率


# ---------- 获取并排序图像 ----------
image_files = [
    f for f in os.listdir(image_folder)
    if f.lower().endswith(('.jpg', '.png'))
]

image_files.sort(key=extract_number)

if not image_files:
    print("No images found in the folder.")
    exit()


# ---------- 读取第一帧 ----------
first_image_path = os.path.join(image_folder, image_files[0])
first_image = cv2.imread(first_image_path)

if first_image is None:
    raise ValueError(f"Failed to read image: {first_image_path}")

height, width = first_image.shape[:2]


# ---------- 创建 VideoWriter（MJPG 高质量） ----------
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

if not video_writer.isOpened():
    raise RuntimeError("Failed to open VideoWriter. Check codec support.")


# ---------- 写入视频 ----------
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"[WARNING] Skip unreadable image: {image_file}")
        continue

    # 只有尺寸不一致才 resize（避免不必要降质）
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height))

    video_writer.write(image)
    print(f"Adding image {image_file} to video.")


# ---------- 释放 ----------
video_writer.release()

print(f"\n✅ Video saved as: {output_video}")