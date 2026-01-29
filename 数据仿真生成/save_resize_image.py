import cv2

# 读取原图
img = cv2.imread("/home/chenkejing/PycharmProjects/ultralytics/images_mode_test/carpet_images_test/34e91fd2aec260eda437b9e567e24df5.jpg")
if img is None:
    raise ValueError("Failed to read input image")

# 直接缩放到 640x640
resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)

# 保存
cv2.imwrite("carpet.jpg", resized)
print("Saved resized image to output_resize.jpg")
