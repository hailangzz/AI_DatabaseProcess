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
        except:
            continue

    return max_idx + 1


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
# 前景贴背景
# =========================

def paste_on_background(bg, fg, mask, min_ratio=0.2, max_ratio=0.8):

    if fg is None or mask is None:
        return bg, None

    Bh, Bw = bg.shape[:2]
    fh, fw = fg.shape[:2]

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

    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

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

def mask_to_yolo_bbox(mask, cls_id):

    h, w = mask.shape

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    x1 = xs.min()
    x2 = xs.max()
    y1 = ys.min()
    y2 = ys.max()

    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h

    return [cls_id, cx, cy, bw, bh]

# =========================
# 主流程
# =========================

def generate_augmented_dataset(
    fg_rgba_dir,
    background_dir,
    out_dir,
    batch_name,
    num_per_object=1,
    cls_id=0
):

    img_out = os.path.join(out_dir, "images")
    labels_root = os.path.join(out_dir, "yolov8_labels")

    seg_out = os.path.join(labels_root, "seg")
    bbox_out = os.path.join(labels_root, "bbox")

    ensure_dir(img_out)
    ensure_dir(seg_out)
    ensure_dir(bbox_out)

    prefix = f"aug_Negative_background_{batch_name}_"
    idx = get_start_index(img_out, prefix)

    bg_imgs = glob(os.path.join(background_dir, "*.jpg"))
    fg_imgs = glob(os.path.join(fg_rgba_dir, "*.png"))

    for fg_path in tqdm(fg_imgs, desc=f"Processing [{batch_name}]"):

        rgba = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)

        if rgba is None or rgba.shape[2] != 4:
            continue

        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]

        mask = (alpha > 10).astype(np.uint8) * 255
        fg = rgb * (mask[..., None] / 255).astype(np.uint8)

        fg, mask = crop_by_mask(fg, mask)

        if fg is None:
            continue

        for _ in range(num_per_object):

            bg = cv2.imread(random.choice(bg_imgs))

            bg, pasted_mask = paste_on_background(bg, fg, mask)

            if pasted_mask is None:
                continue

            seg = mask_to_yolov8_seg(pasted_mask, cls_id)
            bbox = mask_to_yolo_bbox(pasted_mask, cls_id)

            if seg is None or bbox is None:
                continue

            save_name = f"{prefix}{idx:06d}"

            cv2.imwrite(os.path.join(img_out, save_name + ".jpg"), bg)

            # 保存 segmentation
            with open(os.path.join(seg_out, save_name + ".txt"), "w") as f:
                f.write(" ".join([str(seg[0])] + [f"{v:.6f}" for v in seg[1:]]))

            # 保存 bbox
            with open(os.path.join(bbox_out, save_name + ".txt"), "w") as f:
                f.write(" ".join([str(bbox[0])] + [f"{v:.6f}" for v in bbox[1:]]))

            idx += 1

    print(f"✅ 批次 [{batch_name}] 完成，共生成 {idx} 张样本")


# =========================
# 调用
# =========================

if __name__ == "__main__":

    generate_augmented_dataset(
        fg_rgba_dir="/home/chenkejing/database/HandDetect/EmdoorRealHandImages/HandTargetObjectImages",
        background_dir="/home/chenkejing/database/Negativew_Example_Dataset/hand/Negative_hand_database/images",
        out_dir="/home/chenkejing/database/Negativew_Example_Dataset/hand/segment_Negative_hand_database",
        batch_name="origin_real_hand_database",
        num_per_object=1,
        cls_id=0
    )