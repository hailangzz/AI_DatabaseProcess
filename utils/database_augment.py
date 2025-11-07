"""
YOLO-style Data Augmentation Module
Author: zz

Features:
- Random HSV augmentation (hue, saturation, value)
- Random horizontal flip
- Random affine transform (rotation, translation, scale, shear)
- Random resize / letterbox to target size
- Mosaic augmentation (4-image)
- MixUp augmentation
- Cutout (random erasing)
- Utilities to transform bounding boxes accordingly

Dependencies: numpy, cv2, random, math
Optional: torch (for tensors) — functions accept and return numpy arrays; easy to wrap for torch tensors.

Usage example at bottom.
"""

import cv2
import numpy as np
import random
import math
from typing import List, Tuple, Optional


def _clip_bbox(bbox, w, h):
    # bbox: [x_center, y_center, w, h] normalized or absolute depending on caller
    x, y, bw, bh = bbox
    x = max(0, min(x, w))
    y = max(0, min(y, h))
    bw = max(0, min(bw, w))
    bh = max(0, min(bh, h))
    return [x, y, bw, bh]


class YOLOAugmentor:
    def __init__(self,
                 img_size: Tuple[int,int] = (640, 640),
                 hsv_prob: float = 1.0,
                 hsv_gain: Tuple[float,float,float] = (0.015, 0.7, 0.4),
                 flip_prob: float = 0.5,
                 mosaic_prob: float = 0.5,
                 mixup_prob: float = 0.0,
                 cutout_prob: float = 0.0,
                 degrees: float = 10.0,
                 translate: float = 0.1,
                 scale: float = 0.1,
                 shear: float = 2.0,
                 ):
        self.img_h, self.img_w = img_size
        self.hsv_prob = hsv_prob
        self.h_gain, self.s_gain, self.v_gain = hsv_gain
        self.flip_prob = flip_prob
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.cutout_prob = cutout_prob
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    # ----------------------------- Utilities -----------------------------
    def to_numpy(self, img):
        # Accept PIL, torch tensor, or numpy
        if isinstance(img, np.ndarray):
            return img
        try:
            import torch
            if isinstance(img, torch.Tensor):
                img = img.permute(1,2,0).cpu().numpy()
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                return img
        except Exception:
            pass
        raise TypeError('Unsupported image type; provide numpy array or torch tensor')

    @staticmethod
    def bbox_xyxy_to_xywh(bboxes: np.ndarray):
        # bboxes shape (N,4) in xyxy -> xywh
        x1, y1, x2, y2 = bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return np.stack([cx, cy, w, h], axis=1)

    @staticmethod
    def bbox_xywh_to_xyxy(bboxes: np.ndarray):
        # bboxes shape (N,4) in xywh -> xyxy
        cx, cy, w, h = bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.stack([x1,y1,x2,y2], axis=1)

    # ----------------------------- Augmentations -----------------------------
    def random_hsv(self, img: np.ndarray):
        if random.random() > self.hsv_prob:
            return img
        r = np.random.uniform(-1, 1, 3) * np.array([self.h_gain, self.s_gain, self.v_gain])
        hue, sat, val = r
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        img[...,0] = (img[...,0] + hue * 180) % 180
        img[...,1] = img[...,1] * (1 + sat)
        img[...,2] = img[...,2] * (1 + val)
        img[...,1:3] = np.clip(img[...,1:3], 0, 255)
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return img

    def random_flip(self, img: np.ndarray, bboxes: Optional[np.ndarray]=None):
        if random.random() < self.flip_prob:
            img = np.fliplr(img).copy()
            if bboxes is not None and len(bboxes):
                # bboxes: [class, x_center, y_center, w, h] or [x1,y1,x2,y2]
                if bboxes.shape[1] == 5:
                    bboxes[:,1] = img.shape[1] - bboxes[:,1]
                else:
                    x1 = bboxes[:,0].copy()
                    x2 = bboxes[:,2].copy()
                    bboxes[:,0] = img.shape[1] - x2
                    bboxes[:,2] = img.shape[1] - x1
        return img, bboxes

    def random_affine(self, img: np.ndarray, bboxes: Optional[np.ndarray]=None):
        # Based on YOLOv5 random affine implementation
        height = img.shape[0]
        width = img.shape[1]

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(width/2, height/2), scale=s)

        # Shear
        S = np.eye(3)
        sx = math.tan(math.radians(random.uniform(-self.shear, self.shear)))
        sy = math.tan(math.radians(random.uniform(-self.shear, self.shear)))
        S[0,1] = sx
        S[1,0] = sy

        # Translation
        T = np.eye(3)
        tx = random.uniform(-self.translate, self.translate) * width
        ty = random.uniform(-self.translate, self.translate) * height
        T[0,2] = tx
        T[1,2] = ty

        M = T @ S @ R
        imw = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114,114,114))

        if bboxes is None or len(bboxes) == 0:
            return imw, bboxes

        # transform bboxes (xyxy format)
        n = bboxes.shape[0]
        xy = np.ones((n*4,3))
        # x1,y1, x2,y1, x2,y2, x1,y2
        xy[:, :2] = bboxes[:, [0,1,2,1,2,3,0,3]].reshape(n*4,2)
        xy = xy @ M.T
        xy = xy[:, :2].reshape(n,8)
        x = xy[:, [0,2,4,6]]
        y = xy[:, [1,3,5,7]]
        x1 = x.min(1)
        y1 = y.min(1)
        x2 = x.max(1)
        y2 = y.max(1)
        new = np.stack([x1,y1,x2,y2], axis=1)

        # clip
        new[:, [0,2]] = new[:, [0,2]].clip(0, width)
        new[:, [1,3]] = new[:, [1,3]].clip(0, height)
        # filter
        i = (new[:,2] - new[:,0] > 4) & (new[:,3] - new[:,1] > 4)
        bboxes[:, :4] = new
        return imw, bboxes[i]

    def letterbox(self, img: np.ndarray, new_size: Tuple[int,int]=None, color=(114,114,114)):
        if new_size is None:
            new_size = (self.img_h, self.img_w)
        shape = img.shape[:2]  # current shape [h, w]
        ratio = min(new_size[0] / shape[0], new_size[1] / shape[1])
        new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
        dw = new_size[1] - new_unpad[0]
        dh = new_size[0] - new_unpad[1]
        dw //= 2
        dh //= 2
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = dh, new_size[0] - new_unpad[1] - dh
        left, right = dw, new_size[1] - new_unpad[0] - dw
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, ratio, (dw, dh)

    def cutout(self, img: np.ndarray, bboxes: Optional[np.ndarray]=None):
        if random.random() > self.cutout_prob:
            return img, bboxes
        h, w = img.shape[:2]
        num = random.randint(1, 3)
        for _ in range(num):
            ch = random.randint(int(0.02*h), int(0.4*h))
            cw = random.randint(int(0.02*w), int(0.4*w))
            x = random.randint(0, max(0, w - cw))
            y = random.randint(0, max(0, h - ch))
            img[y:y+ch, x:x+cw] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            # optionally remove boxes heavily overlapped - for simplicity not implemented
        return img, bboxes

    def mosaic(self, images: List[np.ndarray], labels: List[np.ndarray]):
        # images: list of 4 images (numpy arrays)
        # labels: list of label arrays in xyxy format [[x1,y1,x2,y2,class], ...]
        s = max(self.img_h, self.img_w)
        yc, xc = [int(random.uniform(int(0.25*s), int(0.75*s))) for _ in range(2)]
        mosaic_img = np.full((s, s, 3), 114, dtype=np.uint8)
        out_labels = []
        for i, (img, label) in enumerate(zip(images, labels)):
            h, w = img.shape[:2]
            # resize
            scale = random.uniform(0.4, 1.0)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
            h, w = img.shape[:2]
            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s), min(s, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)

            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

            pad_x = x1a - x1b
            pad_y = y1a - y1b
            if label.size:
                # label: (N,5) x1,y1,x2,y2,class
                new = label.copy()
                new[:,0] = new[:,0] + pad_x
                new[:,1] = new[:,1] + pad_y
                new[:,2] = new[:,2] + pad_x
                new[:,3] = new[:,3] + pad_y
                out_labels.append(new)
        if len(out_labels):
            out_labels = np.concatenate(out_labels, axis=0)
            # clip
            out_labels[:, [0,2]] = out_labels[:, [0,2]].clip(0, s)
            out_labels[:, [1,3]] = out_labels[:, [1,3]].clip(0, s)
            # filter small
            keep = (out_labels[:,2] - out_labels[:,0] > 4) & (out_labels[:,3] - out_labels[:,1] > 4)
            out_labels = out_labels[keep]
        else:
            out_labels = np.zeros((0,5))
        # resize mosaic to target
        mosaic_img, ratio, (dw, dh) = self.letterbox(mosaic_img, new_size=(self.img_h, self.img_w))
        # adjust labels
        if len(out_labels):
            out_labels[:, [0,2]] = out_labels[:, [0,2]] * ratio + dw
            out_labels[:, [1,3]] = out_labels[:, [1,3]] * ratio + dh
        return mosaic_img, out_labels

    def mixup(self, img1, labels1, img2, labels2, alpha=0.5):
        r = np.random.beta(alpha, alpha) if alpha > 0 else 1
        img = (img1.astype(np.float32) * r + img2.astype(np.float32) * (1 - r)).astype(np.uint8)
        if len(labels1) and len(labels2):
            labels = np.vstack((labels1, labels2))
        elif len(labels1):
            labels = labels1
        elif len(labels2):
            labels = labels2
        else:
            labels = np.zeros((0,5))
        return img, labels

    # ----------------------------- Pipeline -----------------------------
    def augment(self, img: np.ndarray, labels: Optional[np.ndarray]=None):
        """
        img: numpy BGR image
        labels: numpy array shape (N,5) -> x1,y1,x2,y2,class_id (absolute coordinates)
        """
        img = self.to_numpy(img)
        labels = np.array(labels, copy=True) if labels is not None else np.zeros((0,5))

        # Mosaic
        if random.random() < self.mosaic_prob:
            # For simplicity, user must provide 3 additional images via a callback or dataset sampler.
            # Here we just skip mosaic if not enough images.
            return img, labels

        # Random affine
        img, labels = self.random_affine(img, labels)

        # Random flip
        img, labels = self.random_flip(img, labels)

        # HSV
        img = self.random_hsv(img)

        # Cutout
        img, labels = self.cutout(img, labels)

        # Letterbox to size
        img, ratio, (dw, dh) = self.letterbox(img, new_size=(self.img_h, self.img_w))
        if len(labels):
            labels[:, [0,2]] = labels[:, [0,2]] * ratio + dw
            labels[:, [1,3]] = labels[:, [1,3]] * ratio + dh

        return img, labels


# ----------------------------- Example Usage -----------------------------
if __name__ == '__main__':
    # quick test
    aug = YOLOAugmentor(img_size=(640,640), mosaic_prob=0.0, cutout_prob=0.3)
    img = cv2.imread('test.jpg')  # replace with your path
    # sample label: x1,y1,x2,y2,class
    labels = np.array([[50, 50, 200, 200, 0], [300, 100, 400, 250, 1]])
    out_img, out_labels = aug.augment(img, labels)
    print('Out labels:', out_labels)
    cv2.imwrite('aug_out.jpg', out_img)

    # mosaic usage (requires 4 images and labels)
    # imgs = [cv2.imread(p) for p in ['a.jpg','b.jpg','c.jpg','d.jpg']]
    # labs = [np.array(...), ...]
    # mosaic_img, mosaic_labels = aug.mosaic(imgs, labs)
    # cv2.imwrite('mosaic.jpg', mosaic_img)




