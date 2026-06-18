import cv2
import numpy as np


def load_yolo_seg(txt_path, width, height):
    polygons = []

    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines:

        values = line.strip().split()

        cls_id = int(values[0])

        pts = []

        for i in range(1, len(values), 2):
            x = float(values[i]) * width
            y = float(values[i + 1]) * height

            pts.append([x, y])

        pts = np.array(pts, dtype=np.float32)

        polygons.append((cls_id, pts))

    return polygons


def save_yolo_seg(polygons, width, height, txt_path):
    with open(txt_path, "w") as f:

        for cls_id, poly in polygons:

            line = [str(cls_id)]

            for p in poly:
                x = p[0] / width
                y = p[1] / height

                line.append(f"{x:.6f}")
                line.append(f"{y:.6f}")

            f.write(" ".join(line))

            f.write("\n")


def polygon_to_mask(polygon, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)

    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)

    return mask


def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []

    for contour in contours:

        if len(contour) < 3:
            continue

        contour = contour.squeeze(1)

        polygons.append(contour.astype(np.float32))

    return polygons
