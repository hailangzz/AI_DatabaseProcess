import random

import cv2
import numpy as np

from polygon_utils import polygon_to_mask

# ==========================================
# Distance Distribution
# ==========================================

# 1=均匀分布
# 2=偏向远处
# 3=强偏向远处
# 4=极度偏向远处
DISTANCE_BIAS_POWER = 1.0
# 远距离样本比例
DISTANCE_MODE = "uniform"
"""
far: 远距离样本权重大
near: 近距离样本权重大
middle: 中距离样本权重大
uniform: 均匀分布
"""

# ==========================================
# Wire Pixel Length Constraint（生成目标的像素尺寸限制，防止目标像素过小、过大）
# <20px 基本已经很难学习
# 25~50px 正好对应远距离线材
# 50~200px 对应中距离线材
# 200~450px 对应近距离线材
# ==========================================
MIN_PIXEL_LENGTH = 25
MAX_PIXEL_LENGTH = 500

# ==========================================
# Camera Parameters
# ==========================================
CAMERA_FY = 411.535817
CAMERA_CY = 351.293005

CAMERA_HEIGHT = 0.209539
CAMERA_PITCH = 0.0318038


class CopyPasteAugmentor:
    def __init__(self):
        pass

    def sample_wire_length(self):
        """
        生成真实世界线材长度(米)

        分布参考家庭场景:
        50%
            20~50cm

        35%
            50cm~1m

        15%
            1m~2m
        """

        r = random.random()

        if r < 0.50:
            return random.uniform(0.2, 0.35)

        elif r < 0.85:
            return random.uniform(0.35, 0.8)

        else:
            return random.uniform(0.8, 1.2)

    def polygon_length_pixels(self, polygon):

        rect = cv2.minAreaRect(polygon.astype(np.float32))

        w, h = rect[1]

        return max(w, h)

    # ==========================================
    # Pixel Length Filter
    # ==========================================
    def valid_pixel_length(self, polygon):

        pixel_len = self.polygon_length_pixels(polygon)

        if pixel_len < MIN_PIXEL_LENGTH:
            return False

        if pixel_len > MAX_PIXEL_LENGTH:
            return False

        return True

    def compute_length_scale(self, polygon, target_length_m):

        current_pixel_length = self.polygon_length_pixels(polygon)

        if current_pixel_length < 10:
            return 1.0

        PIXELS_PER_METER = 300

        target_pixel_length = target_length_m * PIXELS_PER_METER

        scale = target_pixel_length / current_pixel_length

        return np.clip(scale, 0.1, 2.0)

    def limit_wire_size(self, polygon, image_width):

        pixel_len = self.polygon_length_pixels(polygon)

        max_len = image_width * 0.35

        if pixel_len <= max_len:
            return 1.0

        return max_len / pixel_len

    # ==========================================
    # 图像Y坐标 -> 地面距离(m)
    # ==========================================
    def image_y_to_ground_distance(self, y):

        theta = np.arctan((y - CAMERA_CY) / CAMERA_FY)

        total_theta = theta + CAMERA_PITCH

        if total_theta <= 0:
            return 999.0

        distance = CAMERA_HEIGHT / np.tan(total_theta)

        return distance

    # ==========================================
    # 根据图像位置计算缩放
    # ==========================================
    def perspective_scale(self, y):

        distance = self.image_y_to_ground_distance(y)

        distance = np.clip(distance, 0.15, 2.0)

        scale = (2.0 - distance) / (2.0 - 0.15)

        scale = 0.08 + scale * 1.2

        scale *= random.uniform(0.85, 1.15)

        return scale

    # ==========================================
    # 随机旋转 + 指定缩放
    # ==========================================
    def random_transform(self, rgba, polygon, scale):

        angle = random.uniform(-120, 120)

        h, w = rgba.shape[:2]

        center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, angle, scale)

        cos = abs(M[0, 0])
        sin = abs(M[0, 1])

        new_w = int(h * sin + w * cos)

        new_h = int(h * cos + w * sin)

        M[0, 2] += new_w / 2 - center[0]

        M[1, 2] += new_h / 2 - center[1]

        rgba_new = cv2.warpAffine(
            rgba, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0)
        )

        pts = np.hstack([polygon, np.ones((polygon.shape[0], 1))])

        poly_new = (M @ pts.T).T

        return rgba_new, poly_new

    # ==========================================
    # 优先采样远距离区域
    # ==========================================
    def random_floor_point(self, floor_mask):

        ys, xs = np.where(floor_mask > 0)

        if len(xs) == 0:
            return None

        h = floor_mask.shape[0]

        y_norm = ys.astype(np.float32) / h

        if DISTANCE_MODE == "far":

            # 顶部权重大
            weights = (1.0 - y_norm) ** DISTANCE_BIAS_POWER

        elif DISTANCE_MODE == "near":

            # 底部权重大
            weights = y_norm**DISTANCE_BIAS_POWER

        elif DISTANCE_MODE == "middle":

            # 中间区域权重大
            weights = 1.0 - np.abs(y_norm - 0.5) * 2

            weights = np.clip(weights, 0.01, None)

            weights = weights**DISTANCE_BIAS_POWER

        else:
            # uniform
            weights = np.ones_like(y_norm)

        weights /= weights.sum()

        idx = np.random.choice(len(xs), p=weights)

        return int(xs[idx]), int(ys[idx])

    # ==========================================
    # 平移Polygon
    # ==========================================
    def translate_polygon(self, polygon, dx, dy):

        poly = polygon.copy()

        poly[:, 0] += dx
        poly[:, 1] += dy

        return poly

    # ==========================================
    # Polygon -> Mask
    # ==========================================
    def create_wire_mask(self, polygon, width, height):

        return polygon_to_mask(polygon, width, height)

    # ==========================================
    # 判断是否完全位于地面区域
    # ==========================================
    def inside_floor(self, wire_mask, floor_mask):

        outside = cv2.bitwise_and(wire_mask, cv2.bitwise_not(floor_mask))

        return np.count_nonzero(outside) == 0

    # ==========================================
    # 判断是否与已有线材重叠
    # ==========================================
    def overlap_check(self, wire_mask, placed_mask):

        overlap = cv2.bitwise_and(wire_mask, placed_mask)

        return np.count_nonzero(overlap) > 0

    # ==========================================
    # 添加阴影
    # ==========================================
    def add_shadow(self, bg, wire_mask):

        shadow_mask = cv2.GaussianBlur(wire_mask, (31, 31), 0)

        shadow = shadow_mask.astype(np.float32) * 0.08

        bg = bg.astype(np.float32)

        for c in range(3):
            bg[:, :, c] = np.clip(bg[:, :, c] - shadow, 0, 255)

        return bg.astype(np.uint8)

    # ==========================================
    # Alpha Blend
    # ==========================================
    def alpha_blend(self, bg, rgba, x, y):

        h, w = rgba.shape[:2]

        roi = bg[y : y + h, x : x + w]

        alpha = (rgba[:, :, 3].astype(np.float32) / 255.0)[..., None]

        fg = rgba[:, :, :3]

        blended = fg * alpha + roi * (1 - alpha)

        bg[y : y + h, x : x + w] = blended.astype(np.uint8)

        return bg
