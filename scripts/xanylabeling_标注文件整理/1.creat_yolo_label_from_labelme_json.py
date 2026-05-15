import os
import json


# ==================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def normalize_point(x, y, w, h):
    return x / w, y / h

def polygon_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    bw = xmax - xmin
    bh = ymax - ymin

    return cx, cy, bw, bh

def process_json(json_path, seg_out_path, bbox_out_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]

    seg_lines = []
    bbox_lines = []

    for shape in data["shapes"]:
        label = shape["label"]
        if label not in CLASS_MAP:
            continue

        class_id = CLASS_MAP[label]
        points = shape["points"]

        # ===== seg格式 =====
        seg_line = [str(class_id)]
        for x, y in points:
            nx, ny = normalize_point(x, y, img_w, img_h)
            seg_line.append(f"{nx:.6f}")
            seg_line.append(f"{ny:.6f}")
        seg_lines.append(" ".join(seg_line))

        # ===== bbox格式 =====
        cx, cy, bw, bh = polygon_to_bbox(points)
        cx, cy = normalize_point(cx, cy, img_w, img_h)
        bw /= img_w
        bh /= img_h

        bbox_line = f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        bbox_lines.append(bbox_line)

    # 写文件
    with open(seg_out_path, 'w') as f:
        f.write("\n".join(seg_lines))

    with open(bbox_out_path, 'w') as f:
        f.write("\n".join(bbox_lines))


def main():
    yolov8_label_dir = os.path.join(OUTPUT_DIR, "yolov8_labels")
    seg_dir = os.path.join(yolov8_label_dir, "seg")
    bbox_dir = os.path.join(yolov8_label_dir, "bbox")

    ensure_dir(seg_dir)
    ensure_dir(bbox_dir)

    for file in os.listdir(INPUT_DIR):
        if not file.endswith(".json"):
            continue

        json_path = os.path.join(INPUT_DIR, file)
        name = os.path.splitext(file)[0]

        seg_out = os.path.join(seg_dir, name + ".txt")
        bbox_out = os.path.join(bbox_dir, name + ".txt")

        process_json(json_path, seg_out, bbox_out)

    print("✅ 转换完成！")


# ====== 配置 ======

OUTPUT_DIR = "/data/database/AITotal_Real_Customer_Database/Real_Wire_Customer_Database/date0514/WireSampleFolder"  # 输出目录
INPUT_DIR = os.path.join(OUTPUT_DIR, "images")

# 类别映射（根据你的实际类别修改）
CLASS_MAP = {
    #"liquid": 0,
    # "hand":0,
    #"carpet":0,
    "wire":0
    }

if __name__ == "__main__":
    main()