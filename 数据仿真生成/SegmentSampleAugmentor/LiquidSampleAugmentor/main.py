"""
Wire Copy-Paste Data Augmentation
"""

import os
import random
from datetime import datetime

from tqdm import tqdm

from copy_paste import CopyPasteAugmentor
from polygon_utils import *


def main():
    # =====================================================
    # 创建目录
    # =====================================================

    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

    # =====================================================
    # 初始化
    # =====================================================

    augmentor = CopyPasteAugmentor()

    wire_files = [f for f in os.listdir(Target_DIR) if f.endswith(".png")]

    floor_images = [
        f
        for f in os.listdir(FLOOR_IMG_DIR)
        if os.path.isfile(os.path.join(FLOOR_IMG_DIR, f))
           and f.lower().endswith(IMAGE_EXTS)
    ]

    # =====================================================
    # 开始增强
    # =====================================================

    generated_count = 0

    pbar = tqdm(total=TARGET_NUM_SAMPLES, desc="Augment")

    while generated_count < TARGET_NUM_SAMPLES:

        # 每轮随机打乱背景顺序
        random.shuffle(floor_images)

        for image_name in floor_images:

            if generated_count >= TARGET_NUM_SAMPLES:
                break

            img_path = os.path.join(FLOOR_IMG_DIR, image_name)

            label_path = os.path.join(
                FLOOR_LABEL_DIR,
                os.path.splitext(image_name)[0] + ".txt",
            )

            if not os.path.exists(label_path):
                continue

            bg_origin = cv2.imread(img_path)

            if bg_origin is None:
                continue

            H, W = bg_origin.shape[:2]

            floor_polygons = load_yolo_seg(label_path, W, H)

            if len(floor_polygons) == 0:
                continue

            floor_mask = np.zeros((H, W), dtype=np.uint8)

            for _, poly in floor_polygons:
                floor_mask = cv2.bitwise_or(
                    floor_mask,
                    polygon_to_mask(poly, W, H),
                )

            # =============================================
            # 当前背景生成多张增强图
            # =============================================

            for aug_idx in range(AUG_PER_IMAGE):

                if generated_count >= TARGET_NUM_SAMPLES:
                    break

                bg = bg_origin.copy()

                output_polygons = []

                placed_mask = np.zeros((H, W), dtype=np.uint8)

                num_wire = random.randint(2, 4)

                for _ in range(num_wire):

                    wire_name = random.choice(wire_files)

                    png_path = os.path.join(Target_DIR, wire_name)

                    txt_path = png_path.replace(".png", ".txt")
                    if not os.path.exists(txt_path):
                        print(f"Label not found: {txt_path}")
                        continue

                    rgba = cv2.imread(
                        png_path,
                        cv2.IMREAD_UNCHANGED,
                    )

                    if rgba is None:
                        continue

                    wh, ww = rgba.shape[:2]

                    polys = load_yolo_seg(txt_path, ww, wh)

                    if len(polys) == 0:
                        continue

                    cls_id, poly = polys[0]

                    point = augmentor.random_floor_point(
                        floor_mask
                    )

                    if point is None:
                        continue

                    sample_x, sample_y = point

                    # =====================================
                    # 长度控制
                    # =====================================

                    target_length = (
                        augmentor.sample_wire_length()
                    )

                    length_scale = (
                        augmentor.compute_length_scale(
                            poly,
                            target_length,
                        )
                    )

                    perspective_scale = (
                        augmentor.perspective_scale(
                            sample_y
                        )
                    )

                    scale = (
                            length_scale *
                            perspective_scale
                    )

                    rgba_new, poly_new = (
                        augmentor.random_transform(
                            rgba,
                            poly,
                            scale,
                        )
                    )

                    # =====================================
                    # 长度硬限制
                    # =====================================

                    limit_scale = (
                        augmentor.limit_wire_size(
                            poly_new,
                            W,
                        )
                    )

                    if limit_scale < 1.0:
                        rgba_new, poly_new = (
                            augmentor.random_transform(
                                rgba,
                                poly,
                                scale * limit_scale,
                            )
                        )

                    h, w = rgba_new.shape[:2]

                    success = False

                    for retry in range(100):

                        point = (
                            augmentor.random_floor_point(
                                floor_mask
                            )
                        )

                        if point is None:
                            break

                        x, y = point

                        dx = x - w // 2
                        dy = y - h // 2

                        if (
                                dx < 0
                                or dy < 0
                                or dx + w >= W
                                or dy + h >= H
                        ):
                            continue

                        moved_poly = (
                            augmentor.translate_polygon(
                                poly_new,
                                dx,
                                dy,
                            )
                        )

                        # ===============================
                        # 像素长度过滤
                        # ===============================

                        if not (
                                augmentor.valid_pixel_length(
                                    moved_poly
                                )
                        ):
                            continue

                        wire_mask = np.zeros(
                            (H, W),
                            dtype=np.uint8,
                        )

                        cv2.fillPoly(
                            wire_mask,
                            [moved_poly.astype(np.int32)],
                            255,
                        )

                        # ===============================
                        # 必须全部位于地面
                        # ===============================

                        if not (
                                augmentor.inside_floor(
                                    wire_mask,
                                    floor_mask,
                                )
                        ):
                            continue

                        # ===============================
                        # 不允许重叠
                        # ===============================

                        if (
                                augmentor.overlap_check(
                                    wire_mask,
                                    placed_mask,
                                )
                        ):
                            continue

                        # ===============================
                        # 阴影
                        # ===============================

                        bg = augmentor.add_shadow(
                            bg,
                            wire_mask,
                        )

                        # ===============================
                        # 融合
                        # ===============================

                        bg = augmentor.alpha_blend(
                            bg,
                            rgba_new,
                            dx,
                            dy,
                        )

                        placed_mask = cv2.bitwise_or(
                            placed_mask,
                            wire_mask,
                        )

                        output_polygons.append(
                            (0, moved_poly)
                        )

                        success = True

                        break

                    if not success:
                        continue

                # =============================================
                # 如果一根线都没成功放置，则不保存
                # =============================================

                if len(output_polygons) == 0:
                    continue

                # 整图增强
                bg = augmentor.image_augment(bg)

                # =============================================
                # 保存
                # =============================================

                global sample_number

                file_stem = (
                    f"{CURRENT_DATE}_"
                    f"{MODEL_NAME}_"
                    f"{BATCH_NUMBER}_"
                    f"{sample_number:06d}"
                )

                save_img_name = f"{file_stem}.jpg"

                save_label_name = f"{file_stem}.txt"

                save_img_path = os.path.join(
                    OUTPUT_IMG_DIR,
                    save_img_name,
                )

                save_label_path = os.path.join(
                    OUTPUT_LABEL_DIR,
                    save_label_name,
                )

                cv2.imwrite(save_img_path, bg)

                save_yolo_seg(
                    output_polygons,
                    W,
                    H,
                    save_label_path,
                )

                # 样本编号递增
                sample_number += 1

                generated_count += 1

                pbar.update(1)

    pbar.close()

    print(
        f"Finished. Generated "
        f"{generated_count} samples."
    )


# =====================================================
# 配置
# =====================================================

# 每张背景图生成多少增强图
AUG_PER_IMAGE = 3

# 最终希望生成多少张增强样本
TARGET_NUM_SAMPLES = 5000

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 自定义变量
MODEL_NAME = "liquid"  # 模型名称
BATCH_NUMBER = "public_batch1"  # 批次号

# 获取当前日期
CURRENT_DATE = datetime.now().strftime("%Y%m%d")

# 样本编号起始值
sample_number = 1

if __name__ == "__main__":
    FLOOR_IMG_DIR = "/data/database/Total_Flooring_Images/images"
    FLOOR_LABEL_DIR = "/data/database/Total_Flooring_Images/ground_mask_labels"

    Target_DIR = "/data/database/Total_model_target_mask_png_library/real_image_mask/liquid_mask_png_library"

    output_dir = (
        "/data/database/Total_auto_augmentor_database/liquidDatabaseAugmentor/date0629/real_liquid"
    )
    OUTPUT_IMG_DIR = os.path.join(output_dir, "images")
    OUTPUT_IMG_DIR = os.path.join(output_dir, "images")
    OUTPUT_LABEL_DIR = os.path.join(output_dir, "labels")

    main()
