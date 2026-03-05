import os
import json
from pathlib import Path


def convert_coco_to_yolov8(
        annotations_json: str,
        images_dir: str,
        output_bbox_dir: str,
        output_seg_dir: str
):
    """
    将 COCO 格式标注转换为 YOLOv8 txt 标签

    参数：
        annotations_json: COCO JSON 文件路径
        images_dir: 图片文件夹
        output_bbox_dir: 输出 bbox 标签文件夹
        output_seg_dir: 输出 segmentation 标签文件夹
    """

    os.makedirs(output_bbox_dir, exist_ok=True)
    os.makedirs(output_seg_dir, exist_ok=True)

    # 读取 COCO JSON
    with open(annotations_json, 'r') as f:
        data = json.load(f)

    images = data.get('images', [])
    annotations = data.get('annotations', [])
    categories = {c['id']: c['name'] for c in data.get('categories', [])}

    # 按 image_id 分类标注
    image_to_ann = {}
    for ann in annotations:
        img_id = ann['image_id']
        image_to_ann.setdefault(img_id, []).append(ann)

    # 处理每张图片
    for img_info in images:
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_path = os.path.join(images_dir, file_name)
        base_name = Path(file_name).stem

        if not os.path.exists(img_path):
            print(f"[WARN] 图片不存在，跳过: {img_path}")
            continue

        width, height = img_info['width'], img_info['height']
        anns = image_to_ann.get(img_id, [])

        bbox_lines = []
        seg_lines = []

        for ann in anns:
            class_id = ann['category_id']

            # ---------- YOLOv8 bbox ----------
            if 'bbox' in ann and ann['bbox']:
                x, y, w, h = ann['bbox']  # COCO: x,y,width,height
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w_norm = w / width
                h_norm = h / height
                bbox_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            # ---------- YOLOv8 segmentation ----------
            if 'segmentation' in ann and ann['segmentation']:
                print(ann)
                for seg in ann['segmentation']:
                    if len(seg) < 6:  # 至少三点才能构成多边形
                        continue
                    seg_norm = [f"{seg[i] / width:.6f} {seg[i + 1] / height:.6f}" for i in range(0, len(seg), 2)]
                    seg_line = f"{class_id} " + " ".join(seg_norm)
                    seg_lines.append(seg_line)

        # ---------- 保存 bbox 标签 ----------
        if bbox_lines:
            bbox_file = os.path.join(output_bbox_dir, base_name + ".txt")
            with open(bbox_file, 'w') as f:
                f.write("\n".join(bbox_lines))

        # ---------- 保存 segmentation 标签 ----------
        if seg_lines:
            seg_file = os.path.join(output_seg_dir, base_name + ".txt")
            with open(seg_file, 'w') as f:
                f.write("\n".join(seg_lines))

    print(f"[INFO] 转换完成！bbox 标签保存到 {output_bbox_dir}，seg 标签保存到 {output_seg_dir}")


# --------------------------
# 测试调用
# --------------------------
if __name__ == "__main__":
    convert_coco_to_yolov8(
        annotations_json="/home/chenkejing/database/carpetDatabase/EMdoorRealCarpetDatabase/camera_images_batch1/annotations.json",
        images_dir="/home/chenkejing/database/carpetDatabase/EMdoorRealCarpetDatabase/camera_images_batch1",
        output_bbox_dir="/home/chenkejing/database/carpetDatabase/EMdoorRealCarpetDatabase/camera_images_batch1/yolov8_labels/bbox",
        output_seg_dir="/home/chenkejing/database/carpetDatabase/EMdoorRealCarpetDatabase/camera_images_batch1/yolov8_labels/seg"
    )
