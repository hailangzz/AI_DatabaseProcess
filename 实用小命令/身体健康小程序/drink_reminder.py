# !/usr/bin/env python3
import math
import sys

from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QMouseEvent
from PySide6.QtWidgets import QApplication, QWidget

INTERVAL_MS = 60 * 60 * 1000
SHOW_MS = 120000
TEXT = "💧 该喝水啦！  🚶 请站起来活动5分钟！"


class Reminder(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        geo = QApplication.primaryScreen().geometry()
        self.setGeometry(0, 0, geo.width(), 120)
        self.x = geo.width()
        self.angle = 0
        self.scale = 1.0
        self.hue = 0
        self.visible = False

        self.anim = QTimer(self)
        self.anim.timeout.connect(self.tick)
        self.anim.start(16)

        self.hour = QTimer(self)
        self.hour.timeout.connect(self.show_reminder)
        self.hour.start(INTERVAL_MS)

        self.hide()

    def show_reminder(self):
        self.visible = True
        self.x = self.width()
        self.show()
        QTimer.singleShot(SHOW_MS, self.hide_reminder)

    def hide_reminder(self):
        self.visible = False
        self.hide()

    def mouseDoubleClickEvent(self, e: QMouseEvent):
        self.hide_reminder()

    def tick(self):
        if not self.visible: return
        self.x -= 4
        if self.x < -800:
            self.x = self.width()
        self.angle = math.sin(self.x / 60) * 8
        self.scale = 1 + 0.08 * math.sin(self.x / 40)
        self.hue = (self.hue + 2) % 360
        self.update()

    def paintEvent(self, e):
        if not self.visible: return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(0, 0, 0, 100))
        font = QFont("Sans", 36, QFont.Bold)
        p.setFont(font)
        p.translate(self.x, 60)
        p.rotate(self.angle)
        p.scale(self.scale, self.scale)
        c = QColor()
        c.setHsv(self.hue, 255, 255)
        p.setPen(QPen(c))
        p.drawText(QPointF(0, 0), TEXT)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Reminder()
    # first reminder after 5s for demo
    QTimer.singleShot(5000, w.show_reminder)
    sys.exit(app.exec())
