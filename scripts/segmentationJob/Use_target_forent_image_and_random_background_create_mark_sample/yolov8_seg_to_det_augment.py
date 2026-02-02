import os
import cv2
import random
import numpy as np
from glob import glob
from tqdm import tqdm

# =========================
# 工具函数
# =========================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_start_index(img_out_dir, prefix):
    """
    扫描已有图片文件，返回下一个可用 idx（防止覆盖）
    """
    if not os.path.exists(img_out_dir):
        return 0

    max_idx = -1
    for name in os.listdir(img_out_dir):
        if not name.endswith(".jpg"):
            continue
        if not name.startswith(prefix):
            continue
        try:
            idx = int(name.replace(prefix, "").replace(".jpg", ""))
            max_idx = max(max_idx, idx)
        except ValueError:
            continue

    return max_idx + 1


def read_yolov8_seg_label(txt_path):
    objects = []
    with open(txt_path, "r") as f:
        for line in f:
            nums = list(map(float, line.strip().split()))
            if len(nums) < 7:
                continue
            cls_id = int(nums[0])
            pts = np.array(nums[1:], dtype=np.float32).reshape(-1, 2)
            objects.append((cls_id, pts))
    return objects


def seg_to_mask(points, h, w):
    pts = points.copy()
    pts[:, 0] *= w
    pts[:, 1] *= h
    pts = pts.astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def crop_by_mask(img, mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    if x2 <= x1 or y2 <= y1:
        return None, None
    return img[y1:y2, x1:x2], mask[y1:y2, x1:x2]

# =========================
# 前景贴背景（自动缩放）
# =========================

def paste_on_background(bg, fg, mask, min_ratio=0.2, max_ratio=0.8):
    """
    保证：
    - 前景最短边 / 背景对应边 >= min_ratio
    - 前景最长边 / 背景对应边 <= max_ratio
    """
    if fg is None or mask is None:
        return bg, None

    Bh, Bw = bg.shape[:2]
    fh, fw = fg.shape[:2]

    # 当前比例
    ratio_w = fw / Bw
    ratio_h = fh / Bh
    min_curr = min(ratio_w, ratio_h)
    max_curr = max(ratio_w, ratio_h)

    scale_up = min_ratio / min_curr if min_curr < min_ratio else 1.0
    scale_down = max_ratio / max_curr if max_curr > max_ratio else 1.0

    scale = max(scale_up, scale_down)

    new_w = int(fw * scale)
    new_h = int(fh * scale)

    if new_w <= 0 or new_h <= 0 or new_w >= Bw or new_h >= Bh:
        return bg, None

    fg = cv2.resize(fg, (new_w, new_h))
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8) * 255

    x = random.randint(0, Bw - new_w)
    y = random.randint(0, Bh - new_h)

    alpha = mask[..., None] / 255.0
    bg[y:y+new_h, x:x+new_w] = (
        bg[y:y+new_h, x:x+new_w] * (1 - alpha) + fg * alpha
    ).astype(np.uint8)

    full_mask = np.zeros((Bh, Bw), dtype=np.uint8)
    full_mask[y:y+new_h, x:x+new_w] = mask

    return bg, full_mask

# =========================
# mask -> YOLOv8 seg
# =========================

def mask_to_yolov8_seg(mask, cls_id):
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cnt.shape[0] < 6:
        return None
    cnt = cnt.squeeze(1)
    seg = [cls_id]
    for x, y in cnt:
        seg.append(x / w)
        seg.append(y / h)
    return seg

# =========================
# 主流程
# =========================

def generate_augmented_dataset(
    seg_img_dir,
    seg_label_dir,
    background_dir,
    out_dir,
    batch_name,
    num_per_object=1
):
    img_out = os.path.join(out_dir, "images")
    lbl_out = os.path.join(out_dir, "labels")
    ensure_dir(img_out)
    ensure_dir(lbl_out)

    prefix = f"aug_Negative_background_carpet_{batch_name}_"
    idx = get_start_index(img_out, prefix)

    bg_imgs = glob(os.path.join(background_dir, "*.jpg"))
    src_imgs = glob(os.path.join(seg_img_dir, "*.jpg"))

    for img_path in tqdm(src_imgs, desc=f"Processing [{batch_name}]"):
        name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(seg_label_dir, name + ".txt")
        if not os.path.exists(lbl_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        objects = read_yolov8_seg_label(lbl_path)
        if not objects:
            continue

        best = max(objects, key=lambda x: np.sum(seg_to_mask(x[1], h, w) > 0))
        cls_id, pts = best

        mask = seg_to_mask(pts, h, w)
        fg = img * (mask[..., None] / 255).astype(np.uint8)
        fg, mask = crop_by_mask(fg, mask)
        if fg is None:
            continue

        for _ in range(num_per_object):
            bg = cv2.imread(random.choice(bg_imgs))
            bg, pasted_mask = paste_on_background(bg, fg, mask)
            if pasted_mask is None:
                continue

            seg = mask_to_yolov8_seg(pasted_mask, cls_id)
            if seg is None:
                continue

            save_name = f"{prefix}{idx:06d}"
            cv2.imwrite(os.path.join(img_out, save_name + ".jpg"), bg)
            with open(os.path.join(lbl_out, save_name + ".txt"), "w") as f:
                f.write(" ".join([str(seg[0])] + [f"{v:.6f}" for v in seg[1:]]))

            idx += 1

    print(f"✅ 批次 [{batch_name}] 完成，共生成 {idx} 张样本")

# =========================
# 调用
# =========================

if __name__ == "__main__":

    '''
    1、 /home/chenkejing/database/carpetDatabase/EMdoorRealCarpetDatabase/origin_real_carpet_database
    2、/home/chenkejing/database/carpetDatabase/PublicCarpetDatabase_Myself/origin_public_carpet_database
    3、/home/chenkejing/database/carpetDatabase/PublicCarpetDatabase_Myself/add_images_homeobjects_3k
    '''

    generate_augmented_dataset(
        seg_img_dir="/home/chenkejing/database/carpetDatabase/EMdoorRealCarpetDatabase/origin_real_carpet_database/images",
        seg_label_dir="/home/chenkejing/database/carpetDatabase/EMdoorRealCarpetDatabase/origin_real_carpet_database/labels",
        background_dir="/home/chenkejing/database/Negativew_Example_Dataset/carpet/Negative_carpet_database/images",
        out_dir="/home/chenkejing/database/Negativew_Example_Dataset/carpet/segment_Negative_carpet_database",
        batch_name="origin_real_carpet_database",
        num_per_object=1
    )
