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
image_path = "/地面分割模型应用/绘制地面分割图像demo/real_wire_image_0422_batch3_00240.jpg"

# ==========================================
# 输出目录
# ==========================================
save_dir = "mask2former_result"
os.makedirs(save_dir, exist_ok=True)

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

# RGB格式
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
# 打印所有类别
# ==========================================
print("\nDetected labels:")

unique_ids = np.unique(seg)

for idx in unique_ids:
    label_name = model.config.id2label.get(int(idx), "Unknown")
    print(f"{idx:3d} : {label_name}")

# ==========================================
# 查找 floor 类别
# ==========================================
floor_id = None

for idx, label in model.config.id2label.items():

    label_lower = label.lower()

    if "floor" in label_lower:
        floor_id = idx
        print(f"\nFound floor class: " f"{floor_id} -> {label}")
        break

if floor_id is None:
    print("\nNo floor class found!")

    print("\nAll categories:")

    for idx, label in model.config.id2label.items():
        print(idx, label)

    raise RuntimeError("Cannot find floor category.")

# ==========================================
# 生成地面mask
# ==========================================
floor_mask = np.zeros_like(seg, dtype=np.uint8)

floor_mask[seg == floor_id] = 255

mask_path = os.path.join(save_dir, "floor_mask.png")

cv2.imwrite(mask_path, floor_mask)

print(f"Saved: {mask_path}")

# ==========================================
# 生成绿色覆盖图
# ==========================================
overlay_img = img_rgb.copy()

overlay_img[floor_mask > 0] = [0, 255, 0]

overlay_path = os.path.join(save_dir, "floor_overlay.jpg")

cv2.imwrite(overlay_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

print(f"Saved: {overlay_path}")

# ==========================================
# 半透明覆盖图
# ==========================================
green_mask = np.zeros_like(img_rgb)

green_mask[:, :, 1] = floor_mask

alpha_overlay = cv2.addWeighted(img_rgb, 1.0, green_mask, 0.4, 0)

alpha_path = os.path.join(save_dir, "floor_overlay_alpha.jpg")
cv2.imwrite(alpha_path, cv2.cvtColor(alpha_overlay, cv2.COLOR_RGB2BGR))

print(f"Saved: {alpha_path}")

# ==========================================
# 只保留地面区域
# ==========================================
floor_only = np.zeros_like(img_rgb)

floor_only[floor_mask > 0] = img_rgb[floor_mask > 0]

floor_only_path = os.path.join(save_dir, "floor_only.jpg")

cv2.imwrite(floor_only_path, cv2.cvtColor(floor_only, cv2.COLOR_RGB2BGR))

print(f"Saved: {floor_only_path}")

# ==========================================
# 彩色类别图（调试非常有用）
# ==========================================
np.random.seed(42)

color_map = np.random.randint(0, 255, (256, 3), dtype=np.uint8)

color_seg = color_map[np.clip(seg, 0, 255)]

color_seg_path = os.path.join(save_dir, "semantic_color.png")

cv2.imwrite(color_seg_path, cv2.cvtColor(color_seg, cv2.COLOR_RGB2BGR))

print(f"Saved: {color_seg_path}")

print("\nFinished.")
