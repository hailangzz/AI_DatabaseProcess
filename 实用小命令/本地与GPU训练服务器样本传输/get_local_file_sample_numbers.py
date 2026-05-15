from pathlib import Path


def count_local_files(
    directory,
    suffix_list,
):
    """
    统计指定后缀文件数量
    """

    path = Path(directory)

    count = 0

    for suffix in suffix_list:

        count += len(
            list(
                path.rglob(f"*{suffix}")
            )
        )

    return count


def count_local_dataset(
    image_local_path,
    label_local_path,
):

    print("开始统计本地数据集...\n")

    # 图像后缀
    image_suffixes = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    ]

    # 统计图像数量
    image_count = count_local_files(
        image_local_path,
        image_suffixes
    )

    # 统计txt数量
    txt_count = count_local_files(
        label_local_path,
        [".txt"]
    )

    print("===== 本地数据集统计结果 =====\n")

    print(f"图像路径:")
    print(image_local_path)

    print(f"\n图像数量: {image_count}")

    print("\n-----------------------------\n")

    print(f"标注路径:")
    print(label_local_path)

    print(f"\nTXT数量: {txt_count}")

    print("\n-----------------------------\n")

    # 一致性检查
    if image_count == txt_count:

        print("数据集检查正常：图像数量 与 TXT数量 一致")

    else:

        print("警告：图像数量 与 TXT数量 不一致")

    return image_count, txt_count


if __name__ == "__main__":

    image_local_path = "/data/database/AITotal_SegmentDatabase/wireDatabaseSegment/images/train"
    label_local_path = "/data/database/AITotal_SegmentDatabase/wireDatabaseSegment/labels/train"

    count_local_dataset(
        image_local_path,
        label_local_path,
    )