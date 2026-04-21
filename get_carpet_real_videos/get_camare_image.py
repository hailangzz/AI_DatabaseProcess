"""
# 检查摄像头占用
    lsof /dev/video0
# 关停占用进程
    kill -9 Pid
"""

import cv2
import time
import os

# ---------- 配置 ----------
camera_index = 0                 # 摄像头索引，一般主摄像头为 0
save_dir = "./public_real_camera_images_0417_batch1"     # 保存路径
duration = 15 * 60                # 持续时间，2分钟，单位秒
interval = 0.1                     # 每秒保存10张图像，100毫秒一张
total_frames = duration // interval  # 总帧数，2分钟内每秒保存一帧

# 创建保存目录
os.makedirs(save_dir, exist_ok=True)

# 打开摄像头
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("can not open camera")
    exit()

print(f"start saving images for {duration} seconds, each {interval} seconds...")

start_time = time.time()

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # 保存图片
    # filename = os.path.join(save_dir, f"real_liquid_image_batch5_soy_{frame_count + 1}.jpg")
    filename = os.path.join(save_dir, f"real_liquid_image_batch5_soy_{frame_count + 1:05d}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Saved image: {filename}")

    frame_count += 1

    # 检查是否达到持续时间
    elapsed_time = time.time() - start_time
    if elapsed_time >= duration:
        break

    time.sleep(interval)  # 间隔 1 秒

# 释放摄像头
cap.release()
print("Done!")
