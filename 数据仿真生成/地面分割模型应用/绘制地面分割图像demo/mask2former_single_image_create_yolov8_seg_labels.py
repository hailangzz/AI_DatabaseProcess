import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor
from transformers import Mask2FormerForUniversalSegmentation

# ==========================================
# 模型目录
# ==========================================
model_dir = "/home/chenkejing/Downloads/models/mask2former"

# ==========================================
# 输入图片
# ==========================================
image_path = "test_images/real_wire_image_0422_batch3_00240.jpg"

# ==========================================
# 输出目录
# ==========================================
save_dir = "mask2former_result"
os.makedirs(save_dir, exist_ok=True)

# ==========================================
# 目标类别ID
# ==========================================
TARGET_CLASS_ID = 3

# YOLO类别ID
YOLO_CLASS_ID = 0

# 最小轮廓面积
MIN_AREA = 1000

# ==========================================
# 加载模型
# ==========================================
print("Loading model...")

processor = AutoImageProcessor.from_pretrained(model_dir)

model = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir)

print("Model loaded.")

# ==========================================
# 读取图片
# ==========================================
image = Image.open(image_path).convert("RGB")

img_rgb = np.array(image)

# ==========================================
# 推理
# ==========================================
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# ==========================================
# 获取语义分割结果
# ==========================================
result = processor.post_process_semantic_segmentation(
    outputs, target_sizes=[image.size[::-1]]
)[0]

seg = result.cpu().numpy()

print("Segmentation shape:", seg.shape)

# ==========================================
# 打印检测到的类别
# ==========================================
print("\nDetected labels:")

unique_ids = np.unique(seg)

for idx in unique_ids:
    label_name = model.config.id2label.get(int(idx), "Unknown")
    print(f"{idx}: {label_name}")

# ==========================================
# 提取类别ID=3
# ==========================================
mask = np.zeros_like(seg, dtype=np.uint8)

mask[seg == TARGET_CLASS_ID] = 255

mask_path = os.path.join(save_dir, "class3_mask.png")

cv2.imwrite(mask_path, mask)

print(f"Saved: {mask_path}")

# ==========================================
# 查找轮廓
# ==========================================
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Found {len(contours)} contours")

# ==========================================
# 保存YOLOv8-Seg标签
# ==========================================
h, w = mask.shape

txt_path = os.path.join(save_dir, "class3_yolov8_seg.txt")

saved_count = 0

with open(txt_path, "w") as f:
    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        epsilon = 0.002 * cv2.arcLength(contour, True)

        contour = cv2.approxPolyDP(contour, epsilon, True)

        contour = contour.squeeze(1)

        if len(contour) < 3:
            continue

        line = [str(YOLO_CLASS_ID)]

        for point in contour:
            x = point[0] / w
            y = point[1] / h

            line.append(f"{x:.6f}")

            line.append(f"{y:.6f}")

        f.write(" ".join(line))

        f.write("\n")

        saved_count += 1

print(f"Saved {saved_count} polygons")

print(f"YOLO label saved: {txt_path}")

# ==========================================
# 绘制轮廓
# ==========================================
contour_img = img_rgb.copy()

draw_contours = []

for contour in contours:

    area = cv2.contourArea(contour)

    if area < MIN_AREA:
        continue

    epsilon = 0.002 * cv2.arcLength(contour, True)

    contour = cv2.approxPolyDP(contour, epsilon, True)

    draw_contours.append(contour)

cv2.drawContours(contour_img, draw_contours, -1, (0, 255, 0), 2)

contour_path = os.path.join(save_dir, "class3_contour.jpg")

cv2.imwrite(contour_path, cv2.cvtColor(contour_img, cv2.COLOR_RGB2BGR))

print(f"Saved contour image: {contour_path}")

print("\nFinished.")
