#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
摄像头图像采集程序

功能：
1. 自动检查并结束占用摄像头的进程
2. 定时采集图片
3. Ctrl+C 安全退出
4. 异常退出自动释放摄像头
5. 支持命令行参数

示例：

默认运行
python3 auto_get_camare_images.py

采集10分钟
python3 auto_get_camare_images.py --duration 1

修改保存目录
python3 auto_get_camare_images.py --batch_images_names liquid_batch2

修改摄像头
python3 auto_get_camare_images.py --camera_index 1

组合使用
python3 auto_get_camare_images.py \
    --batch_images_names liquid_batch3 \
    --duration 10 \
    --camera_index 0 \
    --interval 0.1
"""

import argparse
import atexit
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

import cv2

# ======================================================
# 默认参数
# ======================================================

DEFAULT_BATCH_NAME = "real_camera_images_debug_ai_perception"
DEFAULT_DURATION = 1  # 分钟
DEFAULT_INTERVAL = 0.1  # 秒
DEFAULT_CAMERA_INDEX = 0

# ======================================================
# 全局摄像头对象（仅用于退出时释放）
# ======================================================

_cap = None


# ======================================================
# 参数解析
# ======================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Camera Image Collector"
    )

    parser.add_argument(
        "--batch_images_names",
        type=str,
        default=DEFAULT_BATCH_NAME,
        help=f"图片保存目录(默认: {DEFAULT_BATCH_NAME})"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION,
        help=f"采集时长(秒)，默认{DEFAULT_DURATION}"
    )

    parser.add_argument(
        "--camera_index",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help="摄像头编号"
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help="采集间隔(秒)"
    )

    return parser.parse_args()


# ======================================================
# 检查并结束占用摄像头的进程
# ======================================================

def kill_camera_process(camera_index):
    device = f"/dev/video{camera_index}"

    try:
        result = subprocess.run(
            ["lsof", device],
            capture_output=True,
            text=True
        )

        lines = result.stdout.strip().split("\n")

        if len(lines) <= 1:
            print(f"[INFO] {device} 未被占用")
            return

        print(f"[INFO] 检测到 {device} 被以下进程占用：")

        for line in lines[1:]:

            cols = line.split()

            if len(cols) < 2:
                continue

            pid = int(cols[1])
            cmd = cols[0]

            if pid == os.getpid():
                continue

            print(f"    PID={pid} CMD={cmd}")

            try:
                os.kill(pid, signal.SIGKILL)
                print(f"    已结束 PID={pid}")
            except Exception as e:
                print(f"    无法结束 {pid}: {e}")

        time.sleep(1)

    except FileNotFoundError:
        print("系统未安装 lsof")
        print("请执行：")
        print("sudo apt install lsof")


# ======================================================
# 释放摄像头
# ======================================================

def release_camera():
    global _cap

    if _cap is not None:
        _cap.release()
        _cap = None
        print("\n[INFO] 摄像头已释放")


# ======================================================
# Ctrl+C
# ======================================================

def signal_handler(sig, frame):
    print("\n收到退出信号...")
    release_camera()
    cv2.destroyAllWindows()
    sys.exit(0)


# ======================================================
# 主程序
# ======================================================

def main():
    global _cap

    args = parse_args()

    time_str = datetime.now().strftime("%Y%m%d")
    batch_images_names = f"{args.batch_images_names}_{time_str}"
    duration = args.duration * 60
    camera_index = args.camera_index
    interval = args.interval

    save_dir = "./" + batch_images_names

    os.makedirs(save_dir, exist_ok=True)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    atexit.register(release_camera)

    try:

        kill_camera_process(camera_index)

        _cap = cv2.VideoCapture(camera_index)

        if not _cap.isOpened():
            print("无法打开摄像头")
            return

        print("=" * 60)
        print("开始采集")
        print(f"保存目录 : {save_dir}")
        print(f"采集时间 : {duration} 秒")
        print(f"采集间隔 : {interval} 秒")
        print(f"摄像头编号 : {camera_index}")
        print("=" * 60)

        start_time = time.time()

        frame_count = 0

        while True:

            ret, frame = _cap.read()

            if not ret:
                print("读取图像失败")
                break

            image_name = (
                    batch_images_names +
                    f"_{frame_count + 1:05d}.jpg"
            )

            filename = os.path.join(save_dir, image_name)

            # cv2.imwrite(filename, frame)  // 原始保存

            # 压缩到 1280×720
            frame = cv2.resize(
                frame,
                (1280, 720),
                interpolation=cv2.INTER_AREA
            )
            # JPEG质量（可选，95基本无明显损失）
            cv2.imwrite(
                filename,
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )  # // 压缩保存

            frame_count += 1

            print(
                f"\r已保存 {frame_count:05d} 张",
                end="",
                flush=True
            )

            if time.time() - start_time >= duration:
                break

            time.sleep(interval)

        print("\n采集完成！")

    except KeyboardInterrupt:
        print("\n用户终止程序")

    except Exception as e:
        print(f"\n程序异常：{e}")

    finally:
        release_camera()
        cv2.destroyAllWindows()
        print("程序结束")


# ======================================================
# 程序入口
# ======================================================

if __name__ == "__main__":
    main()
