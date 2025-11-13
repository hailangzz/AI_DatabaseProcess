import os, glob
import cv2
import numpy as np

def read_name_list(save_image_direct_path):
    files_name_list = [os.path.basename(f) for f in glob.glob(save_image_direct_path + "/*")]
    return files_name_list

def mark_to_detect(mask_dir): # 根据mask的信息，创建mask目标对应的最小外接矩形（目标检测框）
    # min_area 太小的话，未来标注的标签会特别多，误差很大
    min_area = 100  # 小于该像素面积的目标将被忽略 (测试结果显示，最小像素为100时，检测框标注文件，标注效果很好)
    save_dir = os.path.join(mask_dir[:mask_dir.rfind("/")], "labels")
    os.makedirs(save_dir, exist_ok=True)

    for mask_name in os.listdir(mask_dir):
        if not mask_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue

        mask_path = os.path.join(mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        h, w = mask.shape

        label_path = os.path.join(save_dir, os.path.splitext(mask_name)[0] + ".txt")
        with open(label_path, "w") as f:
            # 找出所有非零类别（跳过背景0）
            for cls_id in np.unique(mask):
                if cls_id == 0:
                    continue

                # 生成该类别的二值掩码
                binary = (mask == cls_id).astype(np.uint8)

                # 查找所有连通区域
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < min_area:
                        continue  # 跳过小目标

                    x, y, bw, bh = cv2.boundingRect(cnt)

                    # 转换为 YOLO 格式（归一化）
                    x_center = (x + bw / 2) / w
                    y_center = (y + bh / 2) / h
                    norm_w = bw / w
                    norm_h = bh / h

                    f.write(f"{int(0)} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    print(f"✅ 转换完成！YOLO标签已保存到：{save_dir}")
    print(f"（已过滤掉面积小于 {min_area} 像素的目标）")

def use_yolo_label_plot_box(image_path):

    image_dir = image_path  # 原始图像文件夹
    label_dir = os.path.join(image_dir[:image_dir.rfind("/")], "labels")  # YOLO标签文件夹
    output_dir = os.path.join(image_dir[:image_dir.rfind("/")], "image_plot_box")  # 输出文件夹
    class_names = ["ElectricWires"]  # 可选: 类别名列表，如 ["person", "car", "dog"]
    os.makedirs(output_dir, exist_ok=True)

    # 颜色生成函数
    def get_color(idx):
        import random
        random.seed(idx)
        return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

    # 遍历所有图片
    for img_name in os.listdir(image_dir):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 无法读取图像：{img_path}")
            continue

        h, w, _ = img.shape

        # 如果没有对应标注文件，跳过
        if not os.path.exists(label_path):
            print(f"⚠️ 未找到标签文件：{label_path}")
            continue

        # 读取标签
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                cls_id, x_center, y_center, bw, bh = map(float, parts)
                cls_id = int(cls_id)

                # 转为像素坐标
                x_center *= w
                y_center *= h
                bw *= w
                bh *= h

                xmin = int(x_center - bw / 2)
                ymin = int(y_center - bh / 2)
                xmax = int(x_center + bw / 2)
                ymax = int(y_center + bh / 2)

                # 颜色与标签名
                color = get_color(cls_id)
                label_text = str(cls_id) if class_names is None else class_names[cls_id]

                # 绘制框
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(img, label_text, (xmin, max(ymin - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 保存可视化结果
        save_path = os.path.join(output_dir, img_name)
        cv2.imwrite(save_path, img)
        print(f"✅ 已保存标注图像：{save_path}")

    print("🎯 全部图像可视化完成！")
    pass


def draw_yolo_boxes(img, boxes,save_path="mosaic_pro.jpg", color=(0, 255, 0), thickness=2):
    """
    在图像上根据 YOLO 格式目标框绘制矩形框

    参数：
        img: numpy.ndarray, 原始图像矩阵 (H, W, C)
        boxes: list[np.ndarray] 或 list[list[float]]
        save_path: 图像保存路径
               YOLO 格式的目标框数组，每个元素为 [cls_id, x_center, y_center, width, height]
        color: tuple(int), 框的颜色 (B, G, R)
        thickness: int, 框线条粗细
    返回：
        绘制了框的图像
    """
    h, w = img.shape[:2]
    img_copy = img.copy()

    for box in boxes:
        cls_id, x_center, y_center, bw, bh = box

        # 转换为像素坐标
        x_center *= w
        y_center *= h
        bw *= w
        bh *= h

        # 计算左上角和右下角坐标
        x1 = int(x_center - bw / 2)
        y1 = int(y_center - bh / 2)
        x2 = int(x_center + bw / 2)
        y2 = int(y_center + bh / 2)
        # 绘制矩形框
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        # 绘制类别文本
        cv2.putText(img_copy, f"ID:{int(cls_id)}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 4)

    cv2.imwrite(save_path, img_copy)

    return img_copy

def resize_long_edge_image(image, target_long=1280):
    """
    按长边缩放图像，并同步调整YOLO标注

    Args:
        image (np.ndarray): 输入图像 (H, W, 3)
        target_long (int): 缩放后的长边尺寸 640、960、1024、1280 （32的倍数）

    Returns:
        resized_img: 缩放后的图像
    """
    h, w = image.shape[:2]
    scale = target_long / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    # 1️⃣ 图像缩放
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return resized