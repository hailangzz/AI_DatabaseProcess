"""
@File    : industrial_yolo_augmentor.py
@Author  : zhangzhuo
@Time    : 2026/03/06
@Description : 工业级YOLO数据增强器，支持Mosaic、MixUp、Cutout、Affine、HSV、Noise、Exposure等
@Version : 1.0
"""
import os
import cv2
import math
import random
import numpy as np
from glob import glob
from typing import Tuple, Optional
from tqdm import tqdm  # 新增进度条库
# ----------------------------- Utility Functions -----------------------------
def xyxy_to_yolo(bboxes: np.ndarray, img_shape: Tuple[int,int]):
    h, w = img_shape[:2]
    if len(bboxes) == 0:
        return np.zeros((0,5))
    x1, y1, x2, y2, cls = bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3], bboxes[:,4]
    x_c = (x1 + x2)/2 / w
    y_c = (y1 + y2)/2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return np.stack([cls, x_c, y_c, bw, bh], axis=1)

def yolo_to_xyxy(bboxes: np.ndarray, img_shape: Tuple[int,int]):
    h, w = img_shape[:2]
    if len(bboxes) == 0:
        return np.zeros((0,5))
    cls, x_c, y_c, bw, bh = bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3], bboxes[:,4]
    x1 = (x_c - bw/2) * w
    y1 = (y_c - bh/2) * h
    x2 = (x_c + bw/2) * w
    y2 = (y_c + bh/2) * h
    return np.stack([x1, y1, x2, y2, cls], axis=1)

def clip_boxes(bboxes: np.ndarray, img_shape: Tuple[int,int]):
    h, w = img_shape[:2]
    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, w)
    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, h)
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, w)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, h)
    return bboxes

# ----------------------------- YOLO Augmentor Class -----------------------------
class IndustrialYOLOAugmentor:
    def __init__(self,
                 img_dir:str,
                 label_dir:str,
                 output_dir:str,
                 batch_name:str="batch",
                 img_size:Tuple[int,int]=(1280,1280),
                 augment_sample_number:int=1000,
                 mosaic_prob:float=0.5,
                 mixup_prob:float=0.2,
                 hsv_prob:float=1.0,
                 flip_prob:float=0.5,
                 cutout_prob:float=0.3,
                 noise_prob:float=0.3,
                 exposure_prob:float=0.3,
                 degrees:float=5.0,
                 translate:float=0.1,
                 scale:float=0.1,
                 shear:float=2.0):

        self.img_dir = img_dir
        self.label_dir = label_dir
        self.output_dir = output_dir
        self.batch_name = batch_name
        self.img_h, self.img_w = img_size
        self.augment_sample_number = augment_sample_number

        # Augmentation params
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.hsv_prob = hsv_prob
        self.flip_prob = flip_prob
        self.cutout_prob = cutout_prob
        self.noise_prob = noise_prob
        self.exposure_prob = exposure_prob
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

        # Load images and labels
        self.img_files = sorted(glob(os.path.join(img_dir, "*.jpg")))
        self.label_files = [os.path.join(label_dir, os.path.basename(x).replace(".jpg",".txt")) for x in self.img_files]

        # Output dirs
        self.save_img_dir = os.path.join(output_dir,"images")
        self.save_label_dir = os.path.join(output_dir,"labels")
        os.makedirs(self.save_img_dir, exist_ok=True)
        os.makedirs(self.save_label_dir, exist_ok=True)

        self.current_index = 0

    # ----------------------------- Load Image & Labels -----------------------------
    def load_image(self, index:int):
        img_path = self.img_files[index]
        label_path = self.label_files[index]
        img = cv2.imread(img_path)
        assert img is not None, f"Image not found: {img_path}"
        labels = []
        if os.path.exists(label_path):
            with open(label_path,'r') as f:
                for line in f.readlines():
                    cls, x, y, w, h = map(float,line.strip().split())
                    labels.append([cls,x,y,w,h])
        return img, np.array(labels,dtype=np.float32)

    # ----------------------------- Augmentations -----------------------------
    def random_hsv(self,img):
        if random.random() > self.hsv_prob:
            return img
        h_gain, s_gain, v_gain = 0.015, 0.7, 0.4
        r = np.random.uniform(-1,1,3)*np.array([h_gain,s_gain,v_gain])
        hue, sat, val = r
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        img[...,0] = (img[...,0] + hue*180) % 180
        img[...,1] = np.clip(img[...,1]*(1+sat),0,255)
        img[...,2] = np.clip(img[...,2]*(1+val),0,255)
        return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def random_flip(self,img,bboxes):
        if random.random() < self.flip_prob:
            img = np.fliplr(img).copy()
            if len(bboxes):
                bboxes[:,1] = 1.0 - bboxes[:,1]
        return img,bboxes

    def random_exposure(self,img):
        if random.random() > self.exposure_prob:
            return img
        alpha = random.uniform(0.5,1.5)
        gamma = random.uniform(0.8,1.2)
        img = np.clip((img/255.0)**gamma*alpha*255.0,0,255).astype(np.uint8)
        return img

    def random_noise(self,img):
        if random.random() > self.noise_prob:
            return img
        h,w,c = img.shape
        noise_type = random.choice(["gaussian","saltpepper"])
        noisy = img.copy()
        if noise_type == "gaussian":
            sigma = random.uniform(10,30)**0.5
            gauss = np.random.normal(0,sigma,(h,w,c))
            noisy = np.clip(img + gauss,0,255).astype(np.uint8)
        else:
            amount = random.uniform(0.002,0.01)
            s_vs_p = 0.5
            num_salt = np.ceil(amount*img.size*s_vs_p)
            num_pepper = np.ceil(amount*img.size*(1-s_vs_p))
            coords = [np.random.randint(0,i-1,int(num_salt)) for i in img.shape[:2]]
            noisy[coords[0],coords[1],:] = 255
            coords = [np.random.randint(0,i-1,int(num_pepper)) for i in img.shape[:2]]
            noisy[coords[0],coords[1],:] = 0
        return noisy

    # def cutout(self,img,bboxes):
    #     if random.random() > self.cutout_prob:
    #         return img,bboxes
    #     h,w = img.shape[:2]
    #     num = random.randint(1,3)
    #     for _ in range(num):
    #         ch = random.randint(int(0.02*h), int(0.4*h))
    #         cw = random.randint(int(0.02*w), int(0.4*w))
    #         x = random.randint(0, max(0,w-cw))
    #         y = random.randint(0, max(0,h-ch))
    #         img[y:y+ch, x:x+cw] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    #     return img,bboxes

    def cutout(self, img, bboxes):
        if random.random() > self.cutout_prob or len(bboxes) == 0:
            return img, bboxes

        h, w = img.shape[:2]
        num = random.randint(1, 3)  # cutout 数量
        for _ in range(num):
            for attempt in range(50):  # 最多尝试50次
                ch = random.randint(int(0.02 * h), int(0.2 * h))
                cw = random.randint(int(0.02 * w), int(0.2 * w))
                x = random.randint(0, max(0, w - cw))
                y = random.randint(0, max(0, h - ch))
                cutout_box = np.array([x, y, x + cw, y + ch])  # xyxy

                # 检查与每个bbox的遮挡比例
                safe = True
                for bbox in yolo_to_xyxy(bboxes, img.shape):
                    x1 = max(cutout_box[0], bbox[0])
                    y1 = max(cutout_box[1], bbox[1])
                    x2 = min(cutout_box[2], bbox[2])
                    y2 = min(cutout_box[3], bbox[3])
                    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if inter_area / bbox_area > 0.2:  # 超过20%则不安全
                        safe = False
                        break
                if safe:
                    img[y:y + ch, x:x + cw] = (random.randint(0, 255),
                                               random.randint(0, 255),
                                               random.randint(0, 255))
                    break
        return img, bboxes

    def random_affine(self,img,bboxes):
        h, w = img.shape[:2]
        if bboxes is None or len(bboxes)==0:
            return img,bboxes

        xyxy = yolo_to_xyxy(bboxes,img.shape)
        a = random.uniform(-self.degrees,self.degrees)
        s = random.uniform(1-self.scale,1+self.scale)
        R = cv2.getRotationMatrix2D((w/2,h/2),a,s)
        S = np.eye(3)
        sx = math.tan(math.radians(random.uniform(-self.shear,self.shear)))
        sy = math.tan(math.radians(random.uniform(-self.shear,self.shear)))
        S[0,1]=sx
        S[1,0]=sy
        T = np.eye(3)
        tx = random.uniform(-self.translate,self.translate)*w
        ty = random.uniform(-self.translate,self.translate)*h
        T[0,2]=tx
        T[1,2]=ty
        M = T@S@np.vstack([R,[0,0,1]])
        n = xyxy.shape[0]
        xy = np.ones((n*4,3))
        xy[:, :2] = xyxy[:, [0,1,2,1,2,3,0,3]].reshape(n*4,2)
        xy = xy@M.T
        xy = xy[:, :2].reshape(n,8)
        new_boxes = np.zeros((n,5))
        for i in range(n):
            pts = xy[i].reshape(4,2)
            x1 = pts[:,0].min()
            y1 = pts[:,1].min()
            x2 = pts[:,0].max()
            y2 = pts[:,1].max()
            new_boxes[i] = [x1,y1,x2,y2,xyxy[i,4]]
        new_boxes = clip_boxes(new_boxes,img.shape)
        w_box = new_boxes[:,2]-new_boxes[:,0]
        h_box = new_boxes[:,3]-new_boxes[:,1]
        keep = (w_box>4)&(h_box>4)
        new_boxes = new_boxes[keep]
        return img, xyxy_to_yolo(new_boxes,img.shape)

    # ----------------------------- Mosaic -----------------------------
    def mosaic_augment(self,index):
        s = max(self.img_h,self.img_w)
        mosaic_img = np.full((s*2,s*2,3),114,dtype=np.uint8)
        yc,xc = [int(random.uniform(s*0.5,s*1.5)) for _ in range(2)]
        indices = [index]+random.choices(range(len(self.img_files)),k=3)
        mosaic_labels=[]
        for i,idx in enumerate(indices):
            img, labels = self.load_image(idx)
            h0,w0 = img.shape[:2]
            scale = s / max(h0,w0)
            nh,nw = int(h0*scale), int(w0*scale)
            img = cv2.resize(img,(nw,nh))
            if i==0:
                x1a,y1a,x2a,y2a = max(xc-nw,0), max(yc-nh,0), xc, yc
                x1b,y1b,x2b,y2b = nw-(x2a-x1a), nh-(y2a-y1a), nw, nh
            elif i==1:
                x1a,y1a,x2a,y2a = xc, max(yc-nh,0), min(xc+nw,s*2), yc
                x1b,y1b,x2b,y2b = 0, nh-(y2a-y1a), min(nw,x2a-x1a), nh
            elif i==2:
                x1a,y1a,x2a,y2a = max(xc-nw,0), yc, xc, min(s*2,yc+nh)
                x1b,y1b,x2b,y2b = nw-(x2a-x1a),0,nw,min(y2a-y1a,nh)
            else:
                x1a,y1a,x2a,y2a = xc,yc,min(xc+nw,s*2),min(s*2,yc+nh)
                x1b,y1b,x2b,y2b = 0,0,min(nw,x2a-x1a),min(nh,y2a-y1a)
            mosaic_img[y1a:y2a,x1a:x2a] = img[y1b:y2b,x1b:x2b]
            if len(labels):
                labels = labels.copy()
                labels[:,1] = labels[:,1]*w0*scale
                labels[:,2] = labels[:,2]*h0*scale
                labels[:,3] = labels[:,3]*w0*scale
                labels[:,4] = labels[:,4]*h0*scale
                labels[:,1] = labels[:,1]-x1b+x1a
                labels[:,2] = labels[:,2]-y1b+y1a
                mosaic_labels.append(labels)
        if len(mosaic_labels):
            mosaic_labels = np.concatenate(mosaic_labels,axis=0)
            xyxy = np.zeros_like(mosaic_labels)
            xyxy[:,0] = mosaic_labels[:,0]
            xyxy[:,1] = mosaic_labels[:,1]-mosaic_labels[:,3]/2
            xyxy[:,2] = mosaic_labels[:,2]-mosaic_labels[:,4]/2
            xyxy[:,3] = mosaic_labels[:,1]+mosaic_labels[:,3]/2
            xyxy[:,4] = mosaic_labels[:,2]+mosaic_labels[:,4]/2
            x1 = max(xc-s//2,0)
            y1 = max(yc-s//2,0)
            x2 = x1+s
            y2 = y1+s
            mosaic_img = mosaic_img[y1:y2,x1:x2]
            xyxy[:,1] = np.clip(xyxy[:,1]-x1,0,s)
            xyxy[:,2] = np.clip(xyxy[:,2]-y1,0,s)
            xyxy[:,3] = np.clip(xyxy[:,3]-x1,0,s)
            xyxy[:,4] = np.clip(xyxy[:,4]-y1,0,s)
            w_box = xyxy[:,3]-xyxy[:,1]
            h_box = xyxy[:,4]-xyxy[:,2]
            keep = (w_box>2)&(h_box>2)
            xyxy = xyxy[keep]
            if len(xyxy):
                labels_new = np.zeros_like(xyxy)
                labels_new[:,0] = xyxy[:,0]
                labels_new[:,1] = (xyxy[:,1]+xyxy[:,3])/2/s
                labels_new[:,2] = (xyxy[:,2]+xyxy[:,4])/2/s
                labels_new[:,3] = w_box[keep]/s
                labels_new[:,4] = h_box[keep]/s
            else:
                labels_new = np.zeros((0,5))
        else:
            labels_new = np.zeros((0,5))
        return mosaic_img, labels_new

    # ----------------------------- Save -----------------------------
    def save_sample(self,img,labels):
        img_path = os.path.join(self.save_img_dir,f"augment_{self.batch_name}_{self.current_index}.jpg")
        label_path = os.path.join(self.save_label_dir,f"augment_{self.batch_name}_{self.current_index}.txt")
        cv2.imwrite(img_path,img)
        with open(label_path,'w') as f:
            for label in labels:
                cls, x, y, w, h = label
                if w<=0 or h<=0 or x<0 or x>1 or y<0 or y>1:
                    continue
                f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        self.current_index += 1

    # ----------------------------- Apply Full Pipeline -----------------------------
    def apply_pipeline(self):
        index = random.randint(0,len(self.img_files)-1)
        if random.random() < self.mosaic_prob:
            img, labels = self.mosaic_augment(index)
        else:
            img, labels = self.load_image(index)
        # img, labels = self.random_affine(img,labels)  # 仿射变换
        img, labels = self.random_flip(img,labels)   # 水平翻转
        img = self.random_hsv(img)      #作用：HSV 色彩增强（Hue 色调、Saturation 饱和度、Value 明亮度）
        img = self.random_exposure(img) # 作用：随机曝光增强（亮度 / gamma 调整）。
        img = self.random_noise(img) #操作：添加噪声（高斯噪声或椒盐噪声）
        img, labels = self.cutout(img,labels) #随机遮挡图像的一部分（黑块 / 彩色块）。
        self.save_sample(img,labels)
        return img, labels

# ----------------------------- Main -----------------------------
if __name__ == "__main__":
    augmentor = IndustrialYOLOAugmentor(
        img_dir="/home/chenkejing/database/HandDetect/EmdoorRealHandImages/train/images",
        label_dir="/home/chenkejing/database/HandDetect/EmdoorRealHandImages/train/labels",
        output_dir="/home/chenkejing/database/HandDetect/EmdoorRealHandImages/database_augmentor",
        batch_name="batch_2",
        augment_sample_number=5500
    )

    # 使用 tqdm 包裹循环，显示进度条
    for _ in tqdm(range(augmentor.augment_sample_number), desc="Augmenting images"):
        img, labels = augmentor.apply_pipeline()
        # 可视化检查
        # for l in labels:
        #     cls,x,y,w,h = l
        #     x1 = int((x-w/2)*1280)
        #     y1 = int((y-h/2)*1280)
        #     x2 = int((x+w/2)*1280)
        #     y2 = int((y+h/2)*1280)
        #     cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        # cv2.imshow("aug",img)
        # cv2.waitKey(1)