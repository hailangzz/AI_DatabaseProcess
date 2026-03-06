import os
import cv2
import numpy as np
import argparse
from scipy.ndimage import map_coordinates, gaussian_filter
from tqdm import tqdm  # 增加进度条

import numpy as np

def add_gaussian_noise(img, mean=0, sigma=10):
    """
    对图像添加高斯噪声，但不对RGBA中alpha=0的区域做处理
    """
    img = img.astype(np.float32)
    noisy = img.copy()

    gauss = np.random.normal(mean, sigma, img.shape).astype(np.float32)

    if img.shape[2] == 4:  # RGBA
        alpha = img[:, :, 3]
        mask = alpha > 0  # 只对非透明区域处理

        for c in range(3):  # 只处理RGB
            channel = noisy[:, :, c]
            channel[mask] = channel[mask] + gauss[:, :, c][mask]
            noisy[:, :, c] = channel

        noisy[:, :, 3] = img[:, :, 3]  # alpha保持不变
    else:
        noisy = img + gauss

    return np.clip(noisy, 0, 255).astype(np.uint8)

def random_rotate(img, max_angle=30):
    h, w = img.shape[:2]
    angle = np.random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

def random_resize(img, min_scale=0.6, max_scale=1.4):
    h, w = img.shape[:2]
    scale = np.random.uniform(min_scale, max_scale)
    nh, nw = int(h*scale), int(w*scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def elastic_transform(image, alpha=30, sigma=5):
    shape = image.shape[:2]
    random_state = np.random.RandomState(None)
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y + dy, x + dx)
    if image.shape[2] == 4:
        channels = [map_coordinates(image[:,:,c], indices, order=1, mode='reflect').reshape(shape) for c in range(4)]
        return np.stack(channels, axis=2).astype(np.uint8)
    else:
        channels = [map_coordinates(image[:,:,c], indices, order=1, mode='reflect').reshape(shape) for c in range(3)]
        return np.stack(channels, axis=2).astype(np.uint8)

def random_brightness(img, brightness_limit=0.3):
    img = img.astype(np.float32)
    beta = np.random.uniform(-brightness_limit * 255, brightness_limit * 255)
    if img.shape[2] == 4:
        img[..., :3] += beta
    else:
        img += beta
    return np.clip(img, 0, 255).astype(np.uint8)

def augment_image(img):
    aug = img.copy()
    aug = random_resize(aug)
    aug = random_rotate(aug)
    aug = elastic_transform(aug)
    aug = add_gaussian_noise(aug)
    aug = random_brightness(aug, brightness_limit=0.3)
    return aug

def process_directory(input_dir, output_dir, n_aug=5):
    os.makedirs(output_dir, exist_ok=True)
    exts = ('.png', '.tif', '.tiff', '.bmp', '.webp')

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(exts)]

    # tqdm 进度条
    for filename in tqdm(files, desc="Processing images", unit="img"):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            print("Failed to read:", img_path)
            continue

        # 确保 RGBA
        if img.shape[2] != 4:
            b,g,r = cv2.split(img)
            a = np.full_like(b, 255)
            img = cv2.merge([b,g,r,a])

        base_name = os.path.splitext(filename)[0]

        for i in range(n_aug):
            aug_img = augment_image(img)
            save_path = os.path.join(output_dir, f"{base_name}_aug{i+1}.png")
            cv2.imencode('.png', aug_img)[1].tofile(save_path)

DEFAULT_INPUT_DIR = "/home/chenkejing/database/HandDetect/EmdoorRealHandImages/train/crop_mark_target_image_results"
DEFAULT_OUTPUT_DIR = "/home/chenkejing/database/HandDetect/EmdoorRealHandImages/HandTargetObjectImages"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGBA 图片数据增强脚本（增加弹性形变 + 进度条）")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR, required=False, help="输入图片目录")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, required=False, help="输出增强图片目录")
    parser.add_argument("--n_aug", type=int, default=9, help="每张图片增强数量")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, args.n_aug)