import os
import random

import cv2
import numpy as np
from tqdm import tqdm


class WireAugmentor:
    def __init__(
            self,
            image_dir,
            label_dir,
            output_dir,
            min_short_side=20,  # 最小短边长度，单位像素，防止生成过小的目标难以学习
    ):

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.output_dir = output_dir

        self.min_short_side = min_short_side

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------
    # 读取YOLOv8 Seg Polygon
    # ------------------------------------------------
    def load_polygon(self, txt_path, w, h):

        polygons = []

        with open(txt_path, "r") as f:

            lines = f.readlines()

        for line in lines:

            values = line.strip().split()

            if len(values) < 7:
                continue

            cls_id = int(values[0])

            pts = []

            for i in range(1, len(values), 2):
                x = float(values[i]) * w
                y = float(values[i + 1]) * h

                pts.append([x, y])

            pts = np.array(pts, dtype=np.float32)

            polygons.append((cls_id, pts))

        return polygons

    # ------------------------------------------------
    # Polygon生成Mask
    # ------------------------------------------------
    def polygon_to_mask(self, polygon, shape):

        mask = np.zeros(shape[:2], dtype=np.uint8)

        cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)

        return mask

    # ------------------------------------------------
    # 目标尺寸过滤
    # ------------------------------------------------
    def is_valid_object(self, polygon):

        _, _, w, h = cv2.boundingRect(polygon.astype(np.int32))

        short_side = min(w, h)

        return short_side >= self.min_short_side

    # ------------------------------------------------
    # 抠图
    # ------------------------------------------------
    def crop_object(self, image, polygon):

        mask = self.polygon_to_mask(polygon, image.shape)

        x, y, w, h = cv2.boundingRect(polygon.astype(np.int32))

        crop_img = image[y: y + h, x: x + w]

        crop_mask = mask[y: y + h, x: x + w]

        crop_poly = polygon.copy()

        crop_poly[:, 0] -= x
        crop_poly[:, 1] -= y

        rgba = cv2.cvtColor(crop_img, cv2.COLOR_BGR2BGRA)

        rgba[:, :, 3] = crop_mask

        return (rgba, crop_poly)

    # ------------------------------------------------
    # 随机旋转缩放
    # ------------------------------------------------
    def random_transform(self, rgba, polygon):

        angle = random.uniform(-180, 180)

        scale = random.uniform(0.3, 1.5)

        h, w = rgba.shape[:2]

        center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, angle, scale)

        cos = abs(M[0, 0])
        sin = abs(M[0, 1])

        new_w = int(h * sin + w * cos)

        new_h = int(h * cos + w * sin)

        M[0, 2] += new_w / 2 - center[0]

        M[1, 2] += new_h / 2 - center[1]

        transformed_img = cv2.warpAffine(
            rgba, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0)
        )

        ones = np.ones((polygon.shape[0], 1), dtype=np.float32)

        pts = np.hstack([polygon, ones])

        transformed_poly = (M @ pts.T).T

        return (transformed_img, transformed_poly)

    # ------------------------------------------------
    # 保存透明PNG
    # ------------------------------------------------
    def save_png(self, rgba, save_path):

        cv2.imwrite(save_path, rgba)

    # ------------------------------------------------
    # 保存Polygon
    # ------------------------------------------------
    def save_polygon(self, polygon, save_path, w, h):

        line = ["0"]

        for p in polygon:
            x = p[0] / w
            y = p[1] / h

            line.append(f"{x:.6f}")

            line.append(f"{y:.6f}")

        with open(save_path, "w") as f:
            f.write(" ".join(line))

    # ------------------------------------------------
    # 主流程
    # ------------------------------------------------
    # ------------------------------------------------
    # 主流程
    # ------------------------------------------------
    def run(self):

        image_names = sorted(
            [
                f
                for f in os.listdir(self.image_dir)
                if f.endswith((".jpg", ".png", ".jpeg"))
            ]
        )

        filtered_count = 0
        saved_count = 0

        for image_name in tqdm(
                image_names,
                desc="Processing Images",
        ):

            try:

                image_path = os.path.join(
                    self.image_dir,
                    image_name,
                )

                label_path = os.path.join(
                    self.label_dir,
                    os.path.splitext(image_name)[0] + ".txt",
                )

                if not os.path.exists(label_path):
                    print(f"Label not found: {label_path}")
                    continue

                image = cv2.imread(image_path)

                if image is None:
                    print(f"Failed to read image: {image_path}")
                    continue

                h, w = image.shape[:2]

                polygons = self.load_polygon(
                    label_path,
                    w,
                    h,
                )

                for idx, (cls_id, polygon) in enumerate(polygons):
                    # -------------------------
                    # 小目标过滤
                    # -------------------------
                    if not self.is_valid_object(polygon):
                        filtered_count += 1
                        continue

                    rgba, crop_poly = self.crop_object(
                        image,
                        polygon,
                    )

                    aug_img, aug_poly = self.random_transform(
                        rgba,
                        crop_poly,
                    )

                    save_name = os.path.splitext(image_name)[0] + f"_{idx}"

                    png_path = os.path.join(
                        self.output_dir,
                        save_name + ".png",
                    )

                    txt_path = os.path.join(
                        self.output_dir,
                        save_name + ".txt",
                    )

                    self.save_png(
                        aug_img,
                        png_path,
                    )

                    hh, ww = aug_img.shape[:2]

                    self.save_polygon(
                        aug_poly,
                        txt_path,
                        ww,
                        hh,
                    )

                    saved_count += 1

            except Exception as e:

                print(f"Error processing {image_name}: {e}")

                continue

        print("\n==============================")
        print(f"Saved Objects    : {saved_count}")
        print(f"Filtered Objects : {filtered_count}")
        print("==============================")


if __name__ == "__main__":
    augmentor = WireAugmentor(
        # image_dir="/data/database/AITotal_Real_Customer_Database/Real_Liquid_Customer_Database/date0616_1/images",
        # label_dir="/data/database/AITotal_Real_Customer_Database/Real_Liquid_Customer_Database/date0616_1/yolov8_labels/seg",
        # # 真实图像mask库
        # output_dir="/data/database/Total_model_target_mask_png_library/real_image_mask/liquid_mask_png_library",
        image_dir="/data/database/LiquadDatabase/TotalLiquidDatabase/images",
        label_dir="/data/database/LiquadDatabase/TotalLiquidDatabase/yolov8_labels/seg",
        # 公开图像mask库
        output_dir="/data/database/Total_model_target_mask_png_library/public_image_mask/liquid_mask_png_library",
        min_short_side=32,
    )

    augmentor.run()
