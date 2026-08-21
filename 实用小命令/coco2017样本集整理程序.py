import json
import os
import random
import shutil

from pycocotools.coco import COCO


def extract_non_person_samples(
        coco_dir: str,
        split: str = "val2017",
        num_samples: int = 100,
        output_dir: str = "./coco_non_person",
        seed: int = 42,
):
    """从 COCO 数据集中提取不含人（person）目标的样本。

    :param coco_dir: COCO 数据集根目录路径 (如 /data/database/coco2017)
    :param split: 数据集划分 ('val2017' 或 'train2017')
    :param num_samples: 准备随机提取的样本数量
    :param output_dir: 结果输出目录
    :param seed: 随机种子，保证抽取结果可复现
    """
    random.seed(seed)

    # 1. 路径准备
    ann_file = os.path.join(
        coco_dir, "annotations", f"instances_{split}.json"
    )
    img_dir = os.path.join(coco_dir, split)

    out_img_dir = os.path.join(output_dir, split)
    out_ann_dir = os.path.join(output_dir, "annotations")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_ann_dir, exist_ok=True)

    print(f"正在加载标注文件: {ann_file}")
    coco = COCO(ann_file)

    # 2. 获取 'person' 类别的 category ID
    person_cat_ids = coco.getCatIds(catNms=["person"])
    if not person_cat_ids:
        raise ValueError("在 COCO 标注中未找到 'person' 类别！")

    # 3. 找出所有包含 'person' 的图片 ID
    person_img_ids = set(coco.getImgIds(catIds=person_cat_ids))

    # 4. 获得所有图片 ID 并剔除包含 'person' 的图片
    all_img_ids = set(coco.getImgIds())
    non_person_img_ids = list(all_img_ids - person_img_ids)

    print(
        f"[{split}] 总图片数: {len(all_img_ids)}, 含人图片数: {len(person_img_ids)}, 不含人图片数: {len(non_person_img_ids)}"
    )

    # 5. 校验请求的抽取数量
    if num_samples > len(non_person_img_ids):
        print(
            f"警告: 请求抽取的数量 ({num_samples}) 大于可用图片数 ({len(non_person_img_ids)})，将提取所有可用图片。"
        )
        selected_img_ids = non_person_img_ids
    else:
        selected_img_ids = random.sample(non_person_img_ids, num_samples)

    print(f"已随机抽取 {len(selected_img_ids)} 张不含人的图片，正在复制图片...")

    # 6. 复制选中的图片文件
    selected_img_objs = coco.loadImgs(selected_img_ids)
    for img_obj in selected_img_objs:
        file_name = img_obj["file_name"]
        src_path = os.path.join(img_dir, file_name)
        dst_path = os.path.join(out_img_dir, file_name)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"警告: 未找到源图片文件 {src_path}")

    # 7. 生成并保存抽取后子集的新 JSON 标注文件
    ann_ids = coco.getAnnIds(imgIds=selected_img_ids)
    selected_anns = coco.loadAnns(ann_ids)

    sub_coco_data = {
        "info": coco.dataset.get("info", {}),
        "licenses": coco.dataset.get("licenses", []),
        "images": selected_img_objs,
        "annotations": selected_anns,
        "categories": coco.dataset.get("categories", []),
    }

    out_ann_file = os.path.join(out_ann_dir, f"instances_{split}.json")
    with open(out_ann_file, "w", encoding="utf-8") as f:
        json.dump(sub_coco_data, f, indent=4)

    print(f"抽取完成！")
    print(f"图片保存在: {out_img_dir}")
    print(f"标注保存在: {out_ann_file}")


if __name__ == "__main__":
    # 配置参数
    COCO_DATASET_DIR = "/data/database/coco2017"  # 你的数据集根目录
    SPLIT = "train2017"  # 提取的目标集合: 'val2017' 或 'train2017'
    NUM_SAMPLES = 15000  # 想要提取的不含人图片数量
    OUTPUT_DIR = "/data/database/coco2017_non_person"  # 输出目录

    extract_non_person_samples(
        coco_dir=COCO_DATASET_DIR,
        split=SPLIT,
        num_samples=NUM_SAMPLES,
        output_dir=OUTPUT_DIR,
        seed=42,  # 更改种子可得到不同抽样结果
    )
