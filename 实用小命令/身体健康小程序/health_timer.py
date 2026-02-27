#!/usr/bin/env python3
import tkinter as tk
import time
import threading
import sys
import math
from screeninfo import get_monitors

# =========================
# 全屏提醒窗口（Insert键关闭）
# =========================
class FullScreenReminder:
    def __init__(self, monitor):
        self.monitor = monitor
        self.root = tk.Tk()
        # 移动窗口到对应显示器
        self.root.geometry(f"{monitor.width}x{monitor.height}+{monitor.x}+{monitor.y}")
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)

        # 背景嫩绿色
        light_green = "#90EE90"
        self.root.configure(bg=light_green)

        canvas = tk.Canvas(
            self.root,
            width=monitor.width,
            height=monitor.height,
            bg=light_green,
            highlightthickness=0
        )
        canvas.pack(fill="both", expand=True)

        # 屏幕中心
        cx, cy = monitor.width // 2, monitor.height // 2

        # 向日葵
        petal_count = 24
        petal_radius = int(monitor.height * 0.125)
        petal_width = int(monitor.height * 0.042)
        petal_height = int(monitor.height * 0.104)
        flower_center_radius = int(monitor.height * 0.0625)

        for i in range(petal_count):
            angle = 2 * math.pi * i / petal_count
            x = cx + petal_radius * math.cos(angle)
            y = cy + petal_radius * math.sin(angle)
            canvas.create_oval(
                x - petal_width // 2,
                y - petal_height // 2,
                x + petal_width // 2,
                y + petal_height // 2,
                fill="#FFD700",
                outline=""
            )

        # 花心
        canvas.create_oval(
            cx - flower_center_radius,
            cy - flower_center_radius,
            cx + flower_center_radius,
            cy + flower_center_radius,
            fill="#8B4513",
            outline=""
        )

        # 提示文字
        canvas.create_text(
            cx,
            cy + int(monitor.height * 0.14),
            text="该活动一下，喝点水 💧\n（请按 Insert 键关闭）",
            fill="white",
            font=("Noto Sans CJK SC", int(monitor.height * 0.028), "bold"),
            justify="center"
        )

        # 仅绑定 Insert 键关闭
        self.root.bind("<Insert>", self.close)

        # 屏蔽其他按键和鼠标点击
        self.root.bind("<Key>", lambda e: None)
        self.root.bind("<Button>", lambda e: None)

    def close(self, event=None):
        self.root.destroy()

    def show(self):
        self.root.mainloop()


# =========================
# 在所有显示器显示提醒
# =========================
def show_reminders_on_all_monitors():
    threads = []
    for monitor in get_monitors():
        t = threading.Thread(target=lambda m=monitor: FullScreenReminder(m).show())
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


# =========================
# 提醒循环
# =========================
def reminder_loop(interval_minutes):
    print(f"健康提醒启动：每 {interval_minutes} 分钟提醒一次")
    while True:
        time.sleep(interval_minutes * 60)
        show_reminders_on_all_monitors()


# =========================
# 选择时间间隔
# =========================
def choose_interval():
    print("\n请选择提醒间隔（分钟）：")
    print("1 - 60 分钟")
    print("2 - 70 分钟（推荐）")
    print("3 - 90 分钟")

    choice_map = {
        "1": 60,  # 测试用
        "2": 70,
        "3": 90
    }

    while True:
        choice = input("请输入 1 / 2 / 3：").strip()
        if choice in choice_map:
            return choice_map[choice]
        else:
            print("输入无效，请重新输入。")


# =========================
# 主入口
# =========================
if __name__ == "__main__":
    try:
        interval = choose_interval()
        t = threading.Thread(
            target=reminder_loop,
            args=(interval,),
            daemon=True
        )
        t.start()

        # 主线程保持运行
        while True:
            time.sleep(3600)

    except KeyboardInterrupt:
        print("\n已退出健康提醒程序")
        sys.exit(0)
