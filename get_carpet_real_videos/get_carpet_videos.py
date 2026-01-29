import cv2
import time

# 打开摄像头（默认是0）
cap = cv2.VideoCapture(0)

# 设置视频编码，使用XVID编码保存为avi格式
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# 创建VideoWriter对象，指定输出文件名，编码器，帧率，分辨率
out = cv2.VideoWriter('output_carpet.avi', fourcc, 20.0, (640, 480))

# 获取当前时间
start_time = time.time()

# 录制5分钟，即300秒
record_duration = 5 * 60  # 5分钟

while True:
    ret, frame = cap.read()
    if not ret:
        print("can not open Capture.")
        break

    # 写入帧到视频文件
    out.write(frame)

    # # 显示摄像头视频流
    # cv2.imshow('frame', frame)

    # 检查录制时间是否超过5分钟
    elapsed_time = time.time() - start_time
    if elapsed_time >= record_duration:
        print("录制时间到达5分钟，自动停止")
        break

    # # 按下'q'键退出
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
