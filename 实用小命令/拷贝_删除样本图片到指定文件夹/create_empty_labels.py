import os


def create_empty_labels(images_dir, labels_dir):
    # 支持的图片格式
    image_exts = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    # 如果labels目录不存在，创建
    os.makedirs(labels_dir, exist_ok=True)

    create_count = 0
    exist_count = 0

    for image_name in os.listdir(images_dir):

        ext = os.path.splitext(image_name)[1].lower()

        # 过滤非图片文件
        if ext not in image_exts:
            continue

        # 获取图片对应的label文件名
        image_stem = os.path.splitext(image_name)[0]

        label_path = os.path.join(
            labels_dir,
            image_stem + ".txt"
        )

        # 不存在则创建空txt
        if not os.path.exists(label_path):

            with open(label_path, "w", encoding="utf-8"):
                pass

            print(
                f"Create empty label: {label_path}"
            )

            create_count += 1

        else:

            exist_count += 1

    print("\n========== Done ==========")
    print(f"Existing labels : {exist_count}")
    print(f"Created labels  : {create_count}")


if __name__ == "__main__":
    images_dir = (
        "/home/chenkejing/Downloads/WireSegmentProject/person_images_yolov11-seg"
    )

    labels_dir = (
        "/home/chenkejing/Downloads/WireSegmentProject/person_detect_yolov11_auto_labels"
    )

    create_empty_labels(
        images_dir,
        labels_dir
    )
