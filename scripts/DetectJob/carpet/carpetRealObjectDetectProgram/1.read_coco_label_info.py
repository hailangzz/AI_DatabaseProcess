import os
import json
import utils.util as util
import numpy as np
from collections import defaultdict

# 读取coco数据集的标注信息、将coco标注转为yolo标注

def read_coco_label_file(coco_label_file_path):
    # 读取 JSON 文件
    with open(coco_label_file_path, "r") as f:
        coco_data = json.load(f)
    # 读取图片信息
    images = coco_data["images"]
    print(f"总图片数: {len(images)}")
    print("第一张图片信息:", images[0])

    # 读取标注信息
    annotations = coco_data["annotations"]
    print(f"总标注数: {len(annotations)}")
    print("第一条标注信息:", annotations[0])

    # 读取类别信息
    categories = coco_data["categories"]
    category_dict = {c['id']: c['name'] for c in categories}
    print("类别字典:", category_dict)


def use_coco_box_info_creat_yolo_label_txt(coco_json_path,
                                           output_dir=r"/home/chenkejing/database/Cable.v1i/test/labels",
                                           target_categories=None):
    image_numbers = 0
    # 输入 COCO 标注文件和图片目录
    global img_info
    coco_json_path = coco_json_path
    output_dir = output_dir

    os.makedirs(output_dir, exist_ok=True)

    # 读取 COCO JSON
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    # 构建类别 id -> 类别索引（YOLO 类索引从 0 开始）
    categories = coco["categories"]
    cat_id_to_index = {cat['id']: idx for idx, cat in enumerate(categories)}

    # 构建 image_id -> image 信息字典
    images = {img['id']: img for img in coco['images']}

    # 将 annotations 按 image_id 分类
    from collections import defaultdict
    image_to_annotations = defaultdict(list)
    print(target_categories)
    for ann in coco['annotations']:

        if target_categories is None:
            image_to_annotations[ann['image_id']].append(ann)
        else:
            if ann["category_id"] not in target_categories:
                continue
            else:
                print(ann)
                image_to_annotations[ann["image_id"]].append(ann)

    # # 遍历每张图片
    # for image_id, img_info in images.items():
    #     img_width = img_info['width']
    #     img_height = img_info['height']
    #
    #     anns = image_to_annotations.get(image_id, [])


    for img_id, anns in image_to_annotations.items():
        img_info = images[img_id]
        img_width, img_height = img_info["width"], img_info["height"]
        yolo_lines = []

        for ann in anns:
            cat_id = ann['category_id']
            bbox = ann['bbox']  # COCO 格式：[x, y, width, height]
            x, y, w, h = bbox

            # 转换为 YOLO 格式 (归一化)
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            w_norm = w / img_width
            h_norm = h / img_height

            class_idx = cat_id_to_index[cat_id]
            yolo_lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        # 输出到 txt 文件
        txt_file = os.path.join(output_dir, os.path.splitext(img_info['file_name'])[0] + ".txt")
        with open(txt_file, "w") as f:
            f.write("\n".join(yolo_lines))

        image_numbers+=1
    print("total images number: ",image_numbers)


def use_coco_segmenta_info_create_yolo_label_txt(
        coco_json_path,
        output_dir,
        target_categories=None,  # 🆕 新增：只处理这些类别
        IOU_THRESHOLD=0.5,
        EXPAND_RATIO=0.45
):
    os.makedirs(output_dir, exist_ok=True)

    # ====== 读取COCO标注 ======
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    cat_id_to_idx = {cat["id"]: i for i, cat in enumerate(coco["categories"])}
    images = {img["id"]: img for img in coco["images"]}

    img_id_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        # 🆕 如果设置了筛选类别，且当前 ann 不在其中，则跳过
        # if target_categories is not None and ann["category_id"] not in target_categories:
        #     continue
        #
        # img_id_to_anns[ann["image_id"]].append(ann)
        if target_categories is None:
            img_id_to_anns[ann["image_id"]].append(ann)
        else:
            if ann["category_id"] not in target_categories:
                continue
            else:
                img_id_to_anns[ann["image_id"]].append(ann)

    # ====== 主循环 ======
    for img_id, anns in img_id_to_anns.items():
        img_info = images[img_id]
        w_img, h_img = img_info["width"], img_info["height"]

        boxes = []
        for ann in anns:
            seg = ann.get("segmentation", [])
            if not seg:
                continue

            # segmentation → 外接矩形
            points = np.concatenate([np.array(s).reshape(-1, 2) for s in seg], axis=0)
            x_min, y_min = points[:, 0].min(), points[:, 1].min()
            x_max, y_max = points[:, 0].max(), points[:, 1].max()

            # 不膨胀版本
            expanded_box = [x_min, y_min, x_max, y_max]

            # 使用 COCO category_id 映射到连续 idx
            cls_id = cat_id_to_idx[ann["category_id"]]
            boxes.append(expanded_box + [cls_id])

        # ====== NMS合并 ======
        merged = []
        while boxes:
            base = boxes.pop(0)
            for b in boxes[:]:
                if b[4] == base[4] and util.iou(base[:4], b[:4]) > IOU_THRESHOLD:
                    base[:4] = util.merge_boxes(base[:4], b[:4])
                    boxes.remove(b)
            merged.append(base)

        # ====== 删除被完全包含的小框 ======
        final_boxes = []
        for i, box_a in enumerate(merged):
            contained = False
            for j, box_b in enumerate(merged):
                if i != j and box_a[4] == box_b[4] and util.is_contained(box_a[:4], box_b[:4]):
                    contained = True
                    break
            if not contained:
                final_boxes.append(box_a)

        # ====== 输出 YOLO 标注 ======
        yolo_lines = []
        for x_min, y_min, x_max, y_max, cls in final_boxes:
            x_center = (x_min + x_max) / 2 / w_img
            y_center = (y_min + y_max) / 2 / h_img
            w_box = (x_max - x_min) / w_img
            h_box = (y_max - y_min) / h_img
            yolo_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w_box:.6f} {h_box:.6f}")

        label_path = os.path.join(output_dir, os.path.splitext(img_info["file_name"])[0] + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        print(f"✅ {img_info['file_name']} -> {len(yolo_lines)} boxes (filtered + merged + cleaned)")

    print("🎯 已完成：COCO segmentation → YOLO（含类别筛选 + NMS + 包裹清理）")


if __name__ == '__main__':
    database_source_path = r"/home/chenkejing/database/carpetDatabase/EMdoorRealCarpetDatabase/camera_images_batch3/"
    database_part_type = "train"

    source_image_path = database_source_path+database_part_type
    target_save_image_path = database_source_path+database_part_type+"/images"
    util.move_batch_image_to_direct(source_image_path, target_save_image_path)

    coco_label_file_path = database_source_path+database_part_type+"/annotations.json"
    read_coco_label_file(coco_label_file_path)

    #
    #
    output_yolo_txt_path = database_source_path+database_part_type+"/labels"
    # 膨胀不膨胀都有问题，膨胀导致原本标注较好的大个大目标，目标框误差增大；不膨胀目标框较多。最终选择不膨胀
    # wire_categories_id_list = [0]
    wire_categories_id_list = [class_id for class_id in range(0, 27)]
    # print("target object classify: ", wire_categories_id_list)
    # wire_categories_id_list = None
    # 分割任务转yolo检测
    use_coco_box_info_creat_yolo_label_txt(coco_label_file_path, output_yolo_txt_path, wire_categories_id_list)
    # use_coco_segmenta_info_create_yolo_label_txt(coco_label_file_path, output_yolo_txt_path, wire_categories_id_list,0.0001, 0.0)
