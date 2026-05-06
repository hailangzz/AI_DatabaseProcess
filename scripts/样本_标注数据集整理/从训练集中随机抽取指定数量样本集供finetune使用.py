import os
import random
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm


# 固定随机种子（保证每次采样一致）
random.seed(42)


def collect_image_label_pairs(image_dir, label_dir):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    pairs = []
    img_suffix = [".jpg", ".jpeg", ".png", ".bmp"]

    for img_path in image_dir.glob("*.*"):
        if img_path.suffix.lower() not in img_suffix:
            continue

        label_path = label_dir / (img_path.stem + ".txt")
        if label_path.exists():
            pairs.append((img_path, label_path))

    return pairs


def copy_pairs(pairs, dst_img_dir, dst_lbl_dir):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    print(f"\n🚚 Copying {len(pairs)} samples...")

    for img_path, lbl_path in tqdm(pairs, desc="Copying", ncols=100):
        shutil.copy2(img_path, dst_img_dir / img_path.name)
        shutil.copy2(lbl_path, dst_lbl_dir / lbl_path.name)


def sample_dataset(src_root, dst_root, num_samples=None, ratio=None):
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    # 只采 train（你原来的逻辑）
    splits = ["train"]

    for split in splits:
        print(f"\n📂 Processing split: {split}")

        img_dir = src_root / "images" / split
        lbl_dir = src_root / "labels" / split

        if not img_dir.exists():
            print(f"⚠️ Skip {split}, path not found")
            continue

        pairs = collect_image_label_pairs(img_dir, lbl_dir)
        total = len(pairs)

        print(f"📊 Total samples: {total}")

        if total == 0:
            print("⚠️ No valid samples found")
            continue

        if num_samples:
            k = min(num_samples, total)
        elif ratio:
            k = int(total * ratio)
        else:
            raise ValueError("Either num_samples or ratio must be set")

        print(f"🎯 Sampling {k} samples...")

        sampled = random.sample(pairs, k)

        dst_img_dir = dst_root / "images" / split
        dst_lbl_dir = dst_root / "labels" / split

        copy_pairs(sampled, dst_img_dir, dst_lbl_dir)

        print(f"✅ Done: {split} → {k} samples copied")


# 默认路径
src_database = "/home/chenkejing/database/AITotal_SegmentDatabase/wireDatabaseSegment"
dst_database = "/data/database/AITotal_SegmentDatabase/finetune_random_sample_datebase/random_wire_database"
number_default = 10000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=src_database, help="source dataset root")
    parser.add_argument("--dst", type=str, default=dst_database, help="destination dataset root")
    parser.add_argument("--num", type=int, default=number_default, help="number of samples per split")
    parser.add_argument("--ratio", type=float, default=None, help="ratio of samples per split")

    args = parser.parse_args()

    print("\n🚀 Start sampling dataset")
    print(f"Source: {args.src}")
    print(f"Target: {args.dst}")

    sample_dataset(
        src_root=args.src,
        dst_root=args.dst,
        num_samples=args.num,
        ratio=args.ratio
    )

    print("\n🎉 All done!")


if __name__ == "__main__":
    main()


"""
拷贝公共训练数据集，到GPU服务器，进行finetune实际训练
scp -r /data/database/AITotal_SegmentDatabase/finetune_random_sample_datebase/random_wire_database   robot-server@172.16.50.229:/home/robot-server/data


"""