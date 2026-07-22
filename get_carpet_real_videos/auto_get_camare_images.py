#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Camera Image Collection Tool

Features:
1. Automatically detect and kill processes occupying the camera
2. Capture images at a fixed interval
3. Safe exit with Ctrl+C
4. Automatically release camera on abnormal exit
5. Support command-line arguments

Examples:

Run with default parameters
python3 auto_get_camare_images.py

Capture for 10 minutes
python3 auto_get_camare_images.py --duration 10

Change output directory
python3 auto_get_camare_images.py --batch_images_names liquid_batch2

Use another camera
python3 auto_get_camare_images.py --camera_index 1

Combined usage
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
# Default parameters
# ======================================================

DEFAULT_BATCH_NAME = "real_camera_images_debug_ai_perception"
DEFAULT_DURATION = 1  # minutes
DEFAULT_INTERVAL = 0.1  # seconds
DEFAULT_CAMERA_INDEX = 0

# ======================================================
# Global camera object
# ======================================================

_cap = None


# ======================================================
# Parse command-line arguments
# ======================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Camera Image Collector"
    )

    parser.add_argument(
        "--batch_images_names",
        type=str,
        default=DEFAULT_BATCH_NAME,
        help=f"Output directory name (default: {DEFAULT_BATCH_NAME})"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION,
        help=f"Capture duration in minutes (default: {DEFAULT_DURATION})"
    )

    parser.add_argument(
        "--camera_index",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help="Camera index"
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help="Capture interval in seconds"
    )

    return parser.parse_args()


# ======================================================
# Kill processes occupying the camera
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
            print(f"[INFO] {device} is not occupied.")
            return

        print(f"[INFO] The following processes are using {device}:")

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
                print(f"    Process {pid} has been terminated.")
            except Exception as e:
                print(f"    Failed to terminate {pid}: {e}")

        time.sleep(1)

    except FileNotFoundError:
        print("lsof is not installed.")
        print("Please install it using:")
        print("sudo apt install lsof")


# ======================================================
# Release camera
# ======================================================

def release_camera():
    global _cap

    if _cap is not None:
        _cap.release()
        _cap = None
        print("\n[INFO] Camera released.")


# ======================================================
# Ctrl+C handler
# ======================================================

def signal_handler(sig, frame):
    print("\nExit signal received...")
    release_camera()
    cv2.destroyAllWindows()
    sys.exit(0)


# ======================================================
# Main function
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
            print("Failed to open camera.")
            return

        print("=" * 60)
        print("Camera image collection started")
        print(f"Output directory : {save_dir}")
        print(f"Duration         : {duration} seconds")
        print(f"Capture interval : {interval} seconds")
        print(f"Camera index     : {camera_index}")
        print("=" * 60)

        start_time = time.time()

        frame_count = 0

        while True:

            ret, frame = _cap.read()

            if not ret:
                print("Failed to capture image.")
                break

            image_name = (
                    batch_images_names +
                    f"_{frame_count + 1:05d}.jpg"
            )

            filename = os.path.join(save_dir, image_name)

            # Resize to 1280×720
            frame = cv2.resize(
                frame,
                (1280, 720),
                interpolation=cv2.INTER_AREA
            )

            # Save as JPEG (quality=95)
            cv2.imwrite(
                filename,
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )

            frame_count += 1

            print(
                f"\rCaptured {frame_count:05d} images",
                end="",
                flush=True
            )

            if time.time() - start_time >= duration:
                break

            time.sleep(interval)

        print("\nImage collection completed.")

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")

    except Exception as e:
        print(f"\nProgram error: {e}")

    finally:
        release_camera()
        cv2.destroyAllWindows()
        print("Program terminated.")


# ======================================================
# Entry point
# ======================================================

if __name__ == "__main__":
    main()
