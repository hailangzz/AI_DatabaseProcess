import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


class Mask2FormerLabelGenerator:
    def __init__(
            self,
            model_dir,
            image_dir,
            label_dir,
            target_class_id=3,
            yolo_class_id=0,
            min_area=1000,
    ):

        self.model_dir = model_dir
        self.image_dir = image_dir
        self.label_dir = label_dir

        self.target_class_id = target_class_id
        self.yolo_class_id = yolo_class_id
        self.min_area = min_area

        os.makedirs(self.label_dir, exist_ok=True)

        self.processor = None
        self.model = None

        # 自动选择设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):

        print(f"Loading model on {self.device}...")

        self.processor = AutoImageProcessor.from_pretrained(self.model_dir)

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_dir)

        self.model.to(self.device)

        self.model.eval()

        # GPU开启半精度
        if self.device.type == "cuda":
            self.model.half()

        print("Model loaded.")

    def get_image_list(self):

        image_files = []

        for file_name in os.listdir(self.image_dir):

            if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                image_files.append(file_name)

        image_files.sort()

        return image_files

    @torch.no_grad()
    def inference_image(self, image_path):

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # GPU FP16推理
        if self.device.type == "cuda":

            with torch.cuda.amp.autocast():

                outputs = self.model(**inputs)

        else:

            outputs = self.model(**inputs)

        result = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]

        return result.cpu().numpy()

    def save_yolov8_seg_label(self, seg, txt_path):

        mask = np.zeros_like(seg, dtype=np.uint8)

        mask[seg == self.target_class_id] = 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = mask.shape

        with open(txt_path, "w") as f:

            for contour in contours:

                area = cv2.contourArea(contour)

                if area < self.min_area:
                    continue

                epsilon = 0.002 * cv2.arcLength(contour, True)

                contour = cv2.approxPolyDP(contour, epsilon, True)

                contour = contour.squeeze(1)

                if len(contour) < 3:
                    continue

                line = [str(self.yolo_class_id)]

                for point in contour:
                    x = point[0] / w
                    y = point[1] / h

                    line.append(f"{x:.6f}")

                    line.append(f"{y:.6f}")

                f.write(" ".join(line))

                f.write("\n")

    def run(self):

        self.load_model()

        image_files = self.get_image_list()

        print(f"Found {len(image_files)} images.")

        start_time = time.time()

        pbar = tqdm(image_files, desc="Mask2Former", unit="img")

        bad_images = []

        for image_name in pbar:

            pbar.set_postfix_str(image_name)

            image_path = os.path.join(self.image_dir, image_name)

            try:

                seg = self.inference_image(image_path)

                txt_name = os.path.splitext(image_name)[0] + ".txt"

                txt_path = os.path.join(self.label_dir, txt_name)

                self.save_yolov8_seg_label(seg, txt_path)

            except Exception as e:

                print(f"\n[ERROR] Failed to process: {image_path}")
                print(f"[ERROR] {e}")

                bad_images.append(image_path)

                continue

        total_time = time.time() - start_time

        fps = len(image_files) / total_time

        print("\nFinished")
        print(f"Images : {len(image_files)}")
        print(f"Time   : {total_time:.2f}s")
        print(f"Speed  : {fps:.2f} img/s")


if __name__ == "__main__":
    # 分割模型目录：用于加载Mask2Former模型权重和配置
    model_dir = "/home/chenkejing/Downloads/models/mask2former"
    # 输入图片目录：包含需要进行地面分割的图像文件
    image_dir = "/data/database/Total_Flooring_Images/images"
    # 输出标签目录：生成的YOLOv8分割标签将保存在此目录下，每个图像对应一个同名的.txt文件
    label_dir = "/data/database/Total_Flooring_Images/ground_mask_labels"

    generator = Mask2FormerLabelGenerator(
        model_dir=model_dir,
        image_dir=image_dir,
        label_dir=label_dir,
        target_class_id=3,
        yolo_class_id=0,
        min_area=1000,
    )

    generator.run()
