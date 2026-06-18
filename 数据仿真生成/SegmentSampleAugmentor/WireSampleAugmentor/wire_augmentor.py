import os
import random

import cv2
import numpy as np


class WireAugmentor:
    def __init__(self, image_dir, label_dir, output_dir):

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.output_dir = output_dir

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
    def run(self):

        image_names = sorted(os.listdir(self.image_dir))

        for image_name in image_names:

            if not image_name.endswith((".jpg", ".png", ".jpeg")):
                continue

            image_path = os.path.join(self.image_dir, image_name)

            label_path = os.path.join(
                self.label_dir, os.path.splitext(image_name)[0] + ".txt"
            )

            image = cv2.imread(image_path)

            h, w = image.shape[:2]

            polygons = self.load_polygon(label_path, w, h)

            for idx, (cls_id, polygon) in enumerate(polygons):
                rgba, crop_poly = self.crop_object(image, polygon)

                (aug_img, aug_poly) = self.random_transform(rgba, crop_poly)

                save_name = os.path.splitext(image_name)[0] + f"_{idx}"

                png_path = os.path.join(self.output_dir, save_name + ".png")

                txt_path = os.path.join(self.output_dir, save_name + ".txt")

                self.save_png(aug_img, png_path)

                hh, ww = aug_img.shape[:2]

                self.save_polygon(aug_poly, txt_path, ww, hh)


if __name__ == "__main__":
    augmentor = WireAugmentor(
        image_dir="/data/database/AITotal_Real_Customer_Database/Real_Wire_Customer_Database/date0514/WireSampleFolder/images",
        label_dir="/data/database/AITotal_Real_Customer_Database/Real_Wire_Customer_Database/date0514/WireSampleFolder/yolov8_labels/seg",
        output_dir="/data/database/Total_model_target_mask_png_library/wire_mask_png_library",
    )

    augmentor.run()
