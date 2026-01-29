import os
import cv2

def gen_hollow_yolov8_seg(
    image_path,
    save_label_path,
    hole_box=(0.4, 0.4, 0.6, 0.6),  # (x1, y1, x2, y2) 归一化
    class_id=0
):
    """
    生成 YOLOv8 segmentation 中空标注
    覆盖整图，中间挖掉一个矩形洞
    """

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图像")

    h, w = img.shape[:2]

    x1, y1, x2, y2 = hole_box

    polygons = []

    # 上
    polygons.append([
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, y1),
        (0.0, y1),
    ])

    # 下
    polygons.append([
        (0.0, y2),
        (1.0, y2),
        (1.0, 1.0),
        (0.0, 1.0),
    ])

    # 左
    polygons.append([
        (0.0, y1),
        (x1, y1),
        (x1, y2),
        (0.0, y2),
    ])

    # 右
    polygons.append([
        (x2, y1),
        (1.0, y1),
        (1.0, y2),
        (x2, y2),
    ])

    os.makedirs(os.path.dirname(save_label_path), exist_ok=True)

    with open(save_label_path, "w") as f:
        for poly in polygons:
            line = [str(class_id)]
            for x, y in poly:
                line.append(f"{x:.6f}")
                line.append(f"{y:.6f}")
            f.write(" ".join(line) + "\n")

    print(f"✔ 中空 YOLOv8-seg 标注已生成: {save_label_path}")


if __name__ == "__main__":
    gen_hollow_yolov8_seg(
        image_path="/home/chenkejing/database/carpetDatabase/PublicCarpetDatabase_Myself/segment_database_augmentor/images/augment_seg_batch1_000042.jpg",
        save_label_path="./image.txt",
        hole_box=(0.4, 0.4, 0.6, 0.6),
        class_id=0
    )
