import os


def clean_yolo_dataset(image_dir, label_dir, image_exts={".jpg", ".png", ".jpeg", ".png"}):
    """
    清理YOLO数据集，只保留image和label都存在的样本
    """

    image_files = {}
    label_files = {}

    # 读取图片
    for f in os.listdir(image_dir):
        name, ext = os.path.splitext(f)
        if ext.lower() in image_exts:
            image_files[name] = f

    # 读取标注
    for f in os.listdir(label_dir):
        name, ext = os.path.splitext(f)
        if ext == ".txt":
            label_files[name] = f

    image_names = set(image_files.keys())
    label_names = set(label_files.keys())

    # 找到不匹配的
    images_without_label = image_names - label_names
    labels_without_image = label_names - image_names

    # 删除没有label的图片
    for name in images_without_label:
        path = os.path.join(image_dir, image_files[name])
        os.remove(path)
        print("删除图片:", path)

    # 删除没有image的label
    for name in labels_without_image:
        path = os.path.join(label_dir, label_files[name])
        os.remove(path)
        print("删除标注:", path)

    print("清理完成")
    print("删除图片:", len(images_without_label))
    print("删除标注:", len(labels_without_image))


if __name__ == "__main__":
    image_dir = "/data/database/LiquadDatabase/spills.v2i.coco/train/imgs"
    label_dir = "/data/database/LiquadDatabase/spills.v2i.coco/train/labels"

    clean_yolo_dataset(image_dir, label_dir)