# -*- coding: utf-8 -*-

"""
JRDB dataset converter

Convert:

jrdb
├── images
│   └── train/val
│       └── image_x
│           └── scene
│               └── xxx.jpg
│
└── labels
    └── train/val
        └── image_x
            └── scene
                └── xxx.txt


To:

jrdb_yolo
├── images
│   ├── train
│   │   └── train_image_0_scene_xxx.jpg
│   └── val
│
└── labels
    ├── train
    └── val

"""

import shutil
from pathlib import Path

from tqdm import tqdm


def build_new_name(
        split,
        camera,
        scene,
        filename
):
    """
    Example:

    train
    image_0
    bytes-cafe-2019-02-07_0
    000123.jpg


    =>

    train_image_0_bytes-cafe-2019-02-07_0_000123.jpg

    """

    stem = Path(filename).stem
    suffix = Path(filename).suffix

    new_name = (
        f"{split}_"
        f"{camera}_"
        f"{scene}_"
        f"{stem}"
        f"{suffix}"
    )

    return new_name


# ============================
# convert one split
# ============================


def convert_split(split):
    print(
        f"\n========== {split} =========="
    )

    image_root = (
            SRC_ROOT /
            "images" /
            split
    )

    label_root = (
            SRC_ROOT /
            "labels" /
            split
    )

    out_image_root = (
            DST_ROOT /
            "images" /
            split
    )

    out_label_root = (
            DST_ROOT /
            "labels" /
            split
    )

    out_image_root.mkdir(
        parents=True,
        exist_ok=True
    )

    out_label_root.mkdir(
        parents=True,
        exist_ok=True
    )

    images = []

    #
    # 找所有图片
    #
    for img_path in image_root.rglob("*"):

        if (
                img_path.is_file()
                and img_path.suffix.lower()
                in IMAGE_EXTS
        ):
            images.append(img_path)

    print(
        f"Found images: {len(images)}"
    )

    missing_labels = 0

    for img_path in tqdm(images):

        relative = (
            img_path
            .relative_to(image_root)
        )

        parts = relative.parts

        #
        # 例如:
        #
        # image_0/
        # bytes-cafe-2019-02-07_0/
        # xxx.jpg
        #

        if len(parts) < 3:
            print(
                "Skip abnormal:",
                img_path
            )

            continue

        camera = parts[0]

        scene = parts[1]

        new_name = build_new_name(
            split,
            camera,
            scene,
            img_path.name
        )

        #
        # copy image
        #

        dst_image = (
                out_image_root /
                new_name
        )

        if not dst_image.exists():
            shutil.copy2(
                img_path,
                dst_image
            )

        #
        # label path
        #

        label_path = (
                label_root /
                relative
        )

        label_path = (
            label_path
            .with_suffix(".txt")
        )

        dst_label = (
                out_label_root /
                Path(new_name)
                .with_suffix(".txt")
        )

        if label_path.exists():

            shutil.copy2(
                label_path,
                dst_label
            )

        else:

            #
            # 没有label生成空文件
            #

            missing_labels += 1

            dst_label.touch()

    print(
        f"Missing labels: {missing_labels}"
    )


# ============================
# Config
# ============================

SRC_ROOT = Path(
    "/data/database/jrdb"
)

DST_ROOT = Path(
    "/data/database/jrdb_yolo"
)

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}

# ============================
# filename builder
# ============================


# ============================
# main
# ============================


if __name__ == "__main__":

    for split in [
        "train",
        "val"
    ]:
        convert_split(
            split
        )

    print(
        "\nDone!"
    )
