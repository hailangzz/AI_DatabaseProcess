"""
@File    : graphic_processing.py
@Author  : zhangzhuo
@Time    : 2025/11/13
@Description : 文件功能简述，例如：
              用于使用mask标注，生成目标检测样本时，扣取目标制作检测样本时，数据增强使用
@Version : 1.0
"""

import os
import cv2
import numpy as np
import random
import utils.util as util

def save_images(output_path, image_data):
    cv2.imwrite(output_path, image_data)


# 根据mask、object等信息，抽取语义分割目标前景图
def extract_object_by_mask_and_yolobox(image_path, mask_path, yolo_label_path):
    """
    根据 mask 和 YOLO 框，从原图中抠出指定目标区域

    Args:
        image_path: 原图路径
        mask_path: 灰度 mask 路径（目标为白，背景黑）
        yolo_label_path: YOLO txt 文件路径（class_id x_center y_center width height）
        output_path: 可选，保存路径
    """
    # 读取原图和mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    image = util.resize_long_edge_image(image,target_long=1280)
    mask = util.resize_long_edge_image(mask,target_long=1280)

    h, w = image.shape[:2]

    # 读取 YOLO 标注（假设每张图只有一个目标）
    with open(yolo_label_path, "r") as f:
        line = f.readline().strip().split()
        _, x_c, y_c, bw, bh = map(float, line)

    # 反归一化
    x_c, y_c, bw, bh = x_c * w, y_c * h, bw * w, bh * h
    x1 = int(x_c - bw / 2)
    y1 = int(y_c - bh / 2)
    x2 = int(x_c + bw / 2)
    y2 = int(y_c + bh / 2)

    # 限制在图像范围内
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # 裁剪原图和mask到目标框区域
    roi_img = image[y1:y2, x1:x2]
    roi_mask = mask[y1:y2, x1:x2]

    # 二值化 mask
    _, roi_mask_bin = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)

    # 抠图
    extracted = cv2.bitwise_and(roi_img, roi_img, mask=roi_mask_bin)

    return extracted


# 对图像进行随机透视变换（仿射+透视效果）
def random_perspective_transform(extract_image, max_warp_ratio=0.2):
    """
    对图像进行随机透视变换（仿射+透视效果）

    Args:
        extract_image: 输入图像（BGR）
        max_warp_ratio: 最大扰动比例（0~1），越大变形越强

    Returns:
        warped_img: 变换后的图像
    """
    image = extract_image
    h, w = image.shape[:2]

    # 原始四个角点
    src = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ])

    # 随机扰动
    def random_shift(val, ratio):
        return val * random.uniform(-ratio, ratio)

    dst = np.float32([
        [0 + random_shift(w, max_warp_ratio), 0 + random_shift(h, max_warp_ratio)],
        [w - 1 + random_shift(w, max_warp_ratio), 0 + random_shift(h, max_warp_ratio)],
        [w - 1 + random_shift(w, max_warp_ratio), h - 1 + random_shift(h, max_warp_ratio)],
        [0 + random_shift(w, max_warp_ratio), h - 1 + random_shift(h, max_warp_ratio)]
    ])

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src, dst)

    # 应用变换
    warped = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return warped


# 获取透视变换后图像的最小外接矩形区域并裁剪
def crop_min_bounding_rect(image, threshold=1):
    """
    截取图像中非零区域的最小外接矩形

    Args:
        image: 输入图像（灰度或彩色 BGR）
        threshold: 非零判定阈值，默认1（像素>0为非零）

    Returns:
        cropped: 裁剪后的图像
        rect: 外接矩形 (x, y, w, h)
    """
    # 如果是彩色图像，转换为灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 创建二值化掩码：非零为1
    mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]

    # 找到非零像素坐标
    coords = cv2.findNonZero(mask)

    if coords is None:
        # 图像全零
        h, w = image.shape[:2]
        return image, (0, 0, w, h)

    # 计算最小外接矩形
    x, y, w, h = cv2.boundingRect(coords)

    # 裁剪图像
    cropped = image[y:y + h, x:x + w]

    return cropped


# 数据增强
def augment_image(image, brightness_range=(0.7, 1.2), contrast_range=(0.7, 1.2), noise_prob=0.3, mask=None):
    """
    对图像进行随机增强：亮度对比度 + 噪点（避免修改无效区域）

    Args:
        image: 输入图像（BGR 或 BGRA）
        brightness_range: 亮度随机范围 (min, max)
        contrast_range: 对比度随机范围 (min, max)
        noise_prob: 噪点概率（0~1）
        mask: 可选，非增强区域为0的mask（如抠图mask）

    Returns:
        aug_img: 增强后的图像（保持原alpha或mask区域）
    """
    img = image.copy().astype(np.float32)

    # 若有 alpha 通道，则提取作为 mask
    if image.shape[2] == 4:
        alpha = image[:, :, 3]
        valid_mask = (alpha > 0).astype(np.uint8)
    elif mask is not None:
        valid_mask = (mask > 0).astype(np.uint8)
    else:
        # 以非全黑像素为有效区域
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        valid_mask = (gray > 10).astype(np.uint8)

    # 亮度与对比度
    alpha_c = random.uniform(*contrast_range)  # 对比度
    beta_b = int((random.uniform(*brightness_range) - 1.0) * 255)  # 亮度偏移
    img_bright = cv2.convertScaleAbs(img, alpha=alpha_c, beta=beta_b)

    # 噪声
    img_noisy = img_bright
    if random.random() < noise_prob:
        noise_type = random.choice(['gaussian', 'salt_pepper'])
        if noise_type == 'gaussian':
            mean = 0
            sigma = random.uniform(5, 20)
            gauss = np.random.normal(mean, sigma, img_bright.shape).astype(np.float32)
            img_noisy = np.clip(img_bright + gauss, 0, 255)
        else:  # 椒盐噪声
            img_noisy = img_bright.copy()
            s_vs_p = 0.5
            amount = random.uniform(0.01, 0.03)
            num_salt = np.ceil(amount * img_noisy.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_noisy.shape[:2]]
            img_noisy[coords[0], coords[1], :] = 255
            num_pepper = np.ceil(amount * img_noisy.size * (1 - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_noisy.shape[:2]]
            img_noisy[coords[0], coords[1], :] = 0

    # 恢复无效区域（保持为原始黑色或透明）
    mask_3c = np.stack([valid_mask]*3, axis=-1)
    img_aug = np.where(mask_3c == 1, img_noisy, img)

    # 若有 alpha 通道，则拼回
    if image.shape[2] == 4:
        img_aug = np.dstack([img_aug.astype(np.uint8), alpha])

    return img_aug.astype(np.uint8)


# 将抠出的目标图像贴到背景图上
def paste_object_on_background(bg_img_path, obj_img, position=None, scale_range=(0.1, 0.2), output_path=None):
    """
    将抠出的目标图像贴到背景图上。

    Args:
        bg_img_path: 背景图路径
        obj_img: 抠出的目标图像（带黑色背景）
        position: (x, y) 左上角位置；若为 None 则随机
        scale_range: (min, max) 随机缩放比例范围
        output_path: 可选，保存路径

    Returns:
        result_img: 合成后的图像
    """

    # 读取背景图
    bg_img = cv2.imread(bg_img_path)
    bh, bw = bg_img.shape[:2]

    # 获取目标尺寸
    oh, ow = obj_img.shape[:2]

    # 随机缩放比例
    scale = random.uniform(scale_range[0], scale_range[1])
    new_w = int(ow * scale)
    new_h = int(oh * scale)

    # 缩放目标图
    obj_resized = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 生成目标mask（黑背景的区域为0）
    obj_gray = cv2.cvtColor(obj_resized, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(obj_gray, 10, 255, cv2.THRESH_BINARY)

    # 若位置未指定，则随机生成
    if position is None:
        max_x = max(bw - new_w, 1)
        max_y = max(bh - new_h, 1)
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
    else:
        x, y = position

    # 确保目标在背景范围内
    if x + new_w > bw or y + new_h > bh:
        raise ValueError("目标超出背景图范围！请调整 position 或缩放。")

    # 创建 ROI 区域
    roi = bg_img[y:y + new_h, x:x + new_w]

    # 反向mask
    mask_inv = cv2.bitwise_not(mask)

    # 背景去除目标区域
    bg_part = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # 目标抠出
    obj_part = cv2.bitwise_and(obj_resized, obj_resized, mask=mask)

    # 合成
    dst = cv2.add(bg_part, obj_part)
    bg_img[y:y + new_h, x:x + new_w] = dst

    return bg_img


# 随机贴图到背景图层上
def paste_A_on_B_with_yolo(A_img, B_img_path, object_class=0, scale_range=(0.1, 0.6)):
    """
    将图像A随机缩放并贴到图像B上，返回YOLO格式的目标框坐标。

    Args:
        A_img_path: 图像A路径（目标）
        B_img_path: 图像B路径（背景）
        scale_range: A相对B的缩放比例范围 (min, max)
        object_class: 目标的检测分类

    Returns:
        result_img: 合成图
        yolo_bbox: [x_center_norm, y_center_norm, width_norm, height_norm]
    """
    # 读取图像
    A = A_img
    B = cv2.imread(B_img_path)
    B = util.resize_long_edge_image(B,target_long=1280) # 为了统一大小，且减小计算量

    if A is None or B is None:
        raise ValueError("无法读取输入图片，请检查路径！")

    Bh, Bw = B.shape[:2]
    Ah, Aw = A.shape[:2]

    # 计算缩放比例（相对背景尺寸）
    scale = random.uniform(scale_range[0], scale_range[1])
    new_w = int(Bw * scale)
    new_h = int(Ah * (new_w / Aw))  # 保持比例
    A_resized = cv2.resize(A, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 随机贴图位置
    max_x = Bw - new_w
    max_y = Bh - new_h
    if max_x <= 0 or max_y <= 0:
        raise ValueError("背景图太小，无法贴入目标！")

    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    # 处理A的透明度或黑背景
    if A_resized.shape[2] == 4:
        # 有alpha通道
        alpha = A_resized[:, :, 3] / 255.0
        alpha = np.stack([alpha] * 3, axis=-1)
        A_rgb = A_resized[:, :, :3]
    else:
        # 无alpha通道，黑背景
        A_rgb = A_resized
        mask = cv2.cvtColor(A_rgb, cv2.COLOR_BGR2GRAY)
        alpha = (mask > 10).astype(np.float32)[..., None]

    # 贴到背景图上
    roi = B[y:y + new_h, x:x + new_w]
    blended = roi * (1 - alpha) + A_rgb * alpha
    B[y:y + new_h, x:x + new_w] = blended.astype(np.uint8)

    # 计算YOLO标注框
    x_center = x + new_w / 2
    y_center = y + new_h / 2
    yolo_bbox = [object_class,
                 x_center / Bw,
                 y_center / Bh,
                 new_w / Bw,
                 new_h / Bh
                 ]

    return B, yolo_bbox
