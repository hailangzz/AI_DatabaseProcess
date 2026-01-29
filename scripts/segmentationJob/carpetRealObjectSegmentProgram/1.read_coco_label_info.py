import os
import json
from pathlib import Path

import numpy as np
import cv2
from pycocotools import mask as maskUtils


def rle_to_polygons(rle, width, height):
    """
    COCO RLE -> polygon list
    返回格式: List[List[x1,y1,x2,y2,...]]
    """
    mask = maskUtils.decode(rle)  # (H, W), uint8

    # OpenCV 找轮廓
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []
    for cnt in contours:
        if cnt.shape[0] < 3:
            continue
        cnt = cnt.reshape(-1, 2)
        poly = cnt.flatten().tolist()
        if len(poly) >= 6:
            polygons.append(poly)

    return polygons


def convert_coco_to_yolov8(
        annotations_json: str,
        images_dir: str,
        output_bbox_dir: str,
        output_seg_dir: str
):
    os.makedirs(output_bbox_dir, exist_ok=True)
    os.makedirs(output_seg_dir, exist_ok=True)

    with open(annotations_json, 'r') as f:
        data = json.load(f)

    images = data.get('images', [])
    annotations = data.get('annotations', [])
    # print(images)
    # print(annotations)

    # category_id -> class_id (0-based)
    categories = sorted([c['id'] for c in data.get('categories', [])])
    cat_id_map = {cid: i for i, cid in enumerate(categories)}


    image_to_ann = {}
    for ann in annotations:
        image_to_ann.setdefault(ann['image_id'], []).append(ann)

    for img_info in images:
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_path = os.path.join(images_dir, file_name)
        base_name = Path(file_name).stem


        if not os.path.exists(img_path):
            continue

        width, height = img_info['width'], img_info['height']
        anns = image_to_ann.get(img_id, [])

        bbox_lines = []
        seg_lines = []

        for ann in anns:
            if ann['category_id'] not in cat_id_map:
                continue

            class_id = cat_id_map[ann['category_id']]

            # ---------- bbox ----------
            if 'bbox' in ann and ann['bbox']:
                x, y, w, h = ann['bbox']
                if w > 0 and h > 0:
                    bbox_lines.append(
                        f"{class_id} "
                        f"{(x + w / 2) / width:.6f} "
                        f"{(y + h / 2) / height:.6f} "
                        f"{w / width:.6f} "
                        f"{h / height:.6f}"
                    )
                    print(ann['bbox'])

            # ---------- segmentation ----------
            polygons = []

            # 1️⃣ polygon 格式
            if isinstance(ann.get('segmentation'), list):
                for seg in ann['segmentation']:
                    if len(seg) >= 6 and len(seg) % 2 == 0:
                        # print(seg)
                        polygons.append(list(map(float, seg)))

            # 2️⃣ RLE 格式（你现在这个）
            elif isinstance(ann.get('segmentation'), dict):
                if ann.get('iscrowd', 0) == 1:
                    rle = ann['segmentation']
                    polygons.extend(
                        rle_to_polygons(rle, width, height)
                    )

            # 写入 YOLOv8
            for poly in polygons:
                seg_norm = []
                for i in range(0, len(poly), 2):
                    seg_norm.append(
                        f"{poly[i] / width:.6f} {poly[i + 1] / height:.6f}"
                    )
                seg_lines.append(f"{class_id} " + " ".join(seg_norm))


        if bbox_lines:
            with open(os.path.join(output_bbox_dir, base_name + ".txt"), 'w') as f:
                print(os.path.join(output_bbox_dir, base_name + ".txt"))
                f.write("\n".join(bbox_lines))

        if seg_lines:
            with open(os.path.join(output_seg_dir, base_name + ".txt"), 'w') as f:
                f.write("\n".join(seg_lines))

    print("[INFO] COCO → YOLOv8 转换完成")


if __name__ == "__main__":
    origin_database = "/home/chenkejing/database/carpetDatabase/rug-pattern-detection.v5-final-version.coco/train/imgs"
    convert_coco_to_yolov8(
        # annotations_json=os.path.join(origin_database, "annotations.json"),
        annotations_json=os.path.join(origin_database, "_annotations.coco.json"),
        images_dir=origin_database,
        output_bbox_dir=os.path.join(origin_database, "yolov8_labels/bbox"),
        output_seg_dir=os.path.join(origin_database, "yolov8_labels/seg")
    )
