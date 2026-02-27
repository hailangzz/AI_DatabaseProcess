import os
import cv2
from ultralytics import YOLO

# ================== 配置 ==================
image_dir = "/home/chenkejing/database/Negativew_Example_Dataset/hand/Negative_hand_coco_database/images"
model_path = "./yolov8n-face.pt"

face_ratio_thresh = 0.25     # 最大人脸面积占整图比例阈值
img_exts = (".jpg", ".jpeg", ".png")
# =========================================

# 加载模型
model = YOLO(model_path)

# 统计信息
total_cnt = 0
keep_cnt = 0
delete_cnt = 0

for img_name in os.listdir(image_dir):
    if not img_name.lower().endswith(img_exts):
        continue

    total_cnt += 1
    img_path = os.path.join(image_dir, img_name)

    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] 读取失败，删除：{img_path}")
        os.remove(img_path)
        delete_cnt += 1
        continue

    h, w = img.shape[:2]
    img_area = h * w

    # 推理
    results = model(img, verbose=False)

    max_face_area = 0

    for r in results:
        if r.boxes is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        for x1, y1, x2, y2 in boxes:
            face_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
            max_face_area = max(max_face_area, face_area)

    # ================== 判断逻辑 ==================
    # 1. 没有人脸 → 保留
    if max_face_area == 0:
        print(f"[KEEP] 无人脸：{img_name}")
        keep_cnt += 1
        continue

    face_ratio = max_face_area / img_area

    # 2. 有人脸但太小 → 删除
    if face_ratio < face_ratio_thresh:
        print(f"[DELETE] 人脸过小 ({face_ratio:.3f})：{img_name}")
        os.remove(img_path)
        delete_cnt += 1
    else:
        # 3. 有人脸且足够大 → 保留
        print(f"[KEEP] 人脸合格 ({face_ratio:.3f})：{img_name}")
        keep_cnt += 1

# ================== 统计汇总 ==================
print("\n========== 清洗完成 ==========")
print(f"总图片数: {total_cnt}")
print(f"保留图片: {keep_cnt}")
print(f"删除图片: {delete_cnt}")
