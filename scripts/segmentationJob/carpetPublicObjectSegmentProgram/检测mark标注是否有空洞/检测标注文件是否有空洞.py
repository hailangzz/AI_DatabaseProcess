import numpy as np
import cv2

def visualize_polygon_and_holes(label_path, img_shape=(1024, 1024)):
    h, w = img_shape

    # 1. 生成 polygon mask
    mask = np.zeros((h, w), dtype=np.uint8)

    with open(label_path, "r") as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            coords = parts[1:]  # 跳过 class id

            pts = np.array([
                [int(coords[i] * w), int(coords[i + 1] * h)]
                for i in range(0, len(coords), 2)
            ], dtype=np.int32)

            cv2.fillPoly(mask, [pts], 255)

    # 2. flood fill 找外部背景
    flood = mask.copy()
    flood = flood.astype(np.uint8)

    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 128)

    # flood == 128 是外部背景
    outside = flood == 128

    # 3. holes = 不属于 polygon，也不属于外部背景
    holes = (mask == 0) & (~outside)

    # 4. 可视化
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[mask == 255] = (0, 255, 0)   # polygon：绿色
    vis[holes] = (0, 0, 255)         # hole：红色

    # 5. 统计信息
    polygon_area = np.sum(mask == 255)
    hole_area = np.sum(holes)

    print(f"polygon 像素数: {polygon_area}")
    print(f"hole 像素数: {hole_area}")

    if hole_area > 0:
        print("✅ 检测到 polygon 内部空洞")
    else:
        print("❌ 未检测到 polygon 内部空洞")

    # 6. 显示
    cv2.imshow("polygon vs holes", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return mask, holes, vis


if __name__ == "__main__":
    # label_path = "./image.txt"
    label_path = "/home/chenkejing/database/carpetDatabase/PublicCarpetDatabase_Myself/segment_database_augmentor/labels/augment_seg_batch1_000008.txt"
    visualize_polygon_and_holes(label_path)
