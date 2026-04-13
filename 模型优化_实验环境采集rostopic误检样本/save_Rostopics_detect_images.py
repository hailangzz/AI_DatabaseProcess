#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
import numpy as np
import cv2
import os
import time

class SaveDetectImages:
    def __init__(self):
        rospy.init_node('save_detect_images_node')

        self.last_trigger_time = 0
        self.latest_image = None

        # 保存路径
        self.save_dir = "/home/chenkejing/PycharmProjects/AI_DatabaseProcess/模型优化_实验环境采集rostopic误检样本/images"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 订阅原始图像（用于保存）
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)

        # 订阅检测图像（用于触发）
        rospy.Subscriber("/camera_detect/object_camera_coordinates_image",
                         Image,
                         self.trigger_callback)

        rospy.loginfo("SaveDetectImages node started.")

    def image_callback(self, msg):
        try:
            # 👉 不用 cv_bridge，直接解析
            img = np.frombuffer(msg.data, dtype=np.uint8)

            if msg.encoding == "rgb8":
                img = img.reshape((msg.height, msg.width, 3))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            elif msg.encoding == "bgr8":
                img = img.reshape((msg.height, msg.width, 3))

            else:
                rospy.logwarn(f"Unsupported encoding: {msg.encoding}")
                return

            self.latest_image = img

        except Exception as e:
            rospy.logerr("Image convert error: %s", str(e))

    def trigger_callback(self, msg):
        # 限制触发频率（1秒最多一张）
        if time.time() - self.last_trigger_time < 1.0:
            return
        self.last_trigger_time = time.time()

        if self.latest_image is None:
            rospy.logwarn("No image cached yet.")
            return

        # 👉 用时间戳命名（不会覆盖）
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.save_dir, f"detect_{timestamp}.jpg")

        try:
            cv2.imwrite(filename, self.latest_image)
            rospy.loginfo(f"Saved image: {filename}")
        except Exception as e:
            rospy.logerr("Save failed: %s", str(e))


if __name__ == "__main__":
    SaveDetectImages()
    rospy.spin()