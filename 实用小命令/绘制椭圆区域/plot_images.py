import cv2
import numpy as np

# 1️⃣ 创建或读取图像
# width, height = 640, 480
# image = np.zeros((height, width, 3), dtype=np.uint8)  # 黑色背景图
# 或者读取图片：
# image = cv2.imread("/home/chenkejing/database/HandDetect/EmdoorRealHandImages/images（复件）/100_1766047898432.jpg")
# image = cv2.imread("/home/chenkejing/Desktop/rgb_images1/4.png")
image = cv2.imread("/home/chenkejing/RKNNProjects/rknn_model_zoo_EMdoor/install/rk3588_linux_aarch64/rknn_yolov8_HandDetectProject_RK3588_demo/model/hand.jpg")

# # 可选：调整图像大小
# resize_width, resize_height = 640, 480  # 你可以根据需要修改这个尺寸
# image = cv2.resize(image, (resize_width, resize_height))
#
# # 2️⃣ 定义椭圆参数
# # center_x, center_y = 540, 340   # 椭圆中心
# # axes_w, axes_h = 310, 315       # 长轴和短轴
# center_x, center_y = 260, 234   # 椭圆中心
# # center_x, center_y = 252, 234   # 椭圆中心
# axes_w, axes_h = 190, 200       # 长轴和短轴

# 可选：调整图像大小
resize_width, resize_height = 1280, 720  # 你可以根据需要修改这个尺寸
image = cv2.resize(image, (resize_width, resize_height))

# 2️⃣ 定义椭圆参数
center_x, center_y = 540, 337   # 椭圆中心
axes_w, axes_h = 290, 300       # 长轴和短轴




angle = 0                       # 椭圆旋转角度
startAngle = 0                   # 起始角度
endAngle = 360                   # 结束角度
color = (0, 255, 0)              # BGR 绿色
thickness = 2                    # 线宽，如果填充用 -1

# 3️⃣ 绘制椭圆
cv2.ellipse(image,
            (center_x, center_y),
            (axes_w, axes_h),
            angle,
            startAngle,
            endAngle,
            color,
            thickness)

# 4️⃣ 显示图像
cv2.imshow("Ellipse Demo", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 5️⃣ 可选：保存图像
cv2.imwrite("ellipse_demo.png", image)
