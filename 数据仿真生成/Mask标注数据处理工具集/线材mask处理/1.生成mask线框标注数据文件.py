import os
import json
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

category_id_dict = {}

def extract_mask_contours(coco_json_path, output_dir):
    """
    coco_json_path: '_annotations.coco.json' 文件路径
    output_dir: 输出 mask 轮廓文件目录，每张图片一个 JSON
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载 COCO 注释
    coco = COCO(coco_json_path)

    # 遍历每张图片
    for img_id in coco.imgs:
        img_info = coco.imgs[img_id]
        file_name = img_info['file_name']
        img_area = img_info['height'] * img_info['width']

        # 获取该图片的所有 annotation
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        contours_per_image = []

        for ann in anns:
            seg = ann['segmentation']

            # 计算 mask 面积
            if isinstance(seg, list):
                # Polygon -> 转 RLE
                rles = maskUtils.frPyObjects(seg, img_info['height'], img_info['width'])
                rle = maskUtils.merge(rles)
            else:
                rle = seg
            area = maskUtils.area(rle)

            # 判断 category_id 且 mask 面积大于图片一半
            if ann["category_id"] == 1 and area >= img_area / 5:
                polygons = []
                if isinstance(seg, list):
                    for poly in seg:
                        polygons.append(poly)
                else:
                    # RLE -> Polygon
                    from skimage import measure
                    mask = maskUtils.decode(rle)
                    contours = measure.find_contours(mask, 0.5)
                    for contour in contours:
                        contour = contour[:, [1,0]].flatten().tolist()  # xy 翻转
                        polygons.append(contour)

                contours_per_image.append({
                    "category_id": ann["category_id"],
                    "polygons": polygons,
                    "area": int(area)
                })

                if ann["category_id"] not in category_id_dict:
                    category_id_dict[ann["category_id"]] = 1
                else:
                    category_id_dict[ann["category_id"]] += 1

        # 只有当有满足条件的标注时才写文件
        if contours_per_image:
            output_file = os.path.splitext(file_name)[0] + ".json"
            with open(os.path.join(output_dir, output_file), "w") as f:
                json.dump(contours_per_image, f)
            print(f"Saved contours: {output_file}")
        else:
            print(f"Skipped {file_name}: no large enough mask")

if __name__ == "__main__":
    coco_json_path = "/home/chenkejing/database/Floor/floor.v1i.coco/train/_annotations.coco.json"
    output_dir = coco_json_path.split("train/")[0] + "mask_contours"
    extract_mask_contours(coco_json_path, output_dir)
    print(category_id_dict)
