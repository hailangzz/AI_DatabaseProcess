import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class LabelChecker(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("目标检测标注检查器")
        self.resize(1100, 750)

        self.image_dir = ""
        self.label_dir = ""
        self.files = []
        self.index = 0
        self.batch_size = 10  # 每页显示图片数量

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # 路径输入区
        path_layout = QHBoxLayout()
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setPlaceholderText("图片文件夹路径")
        self.label_path_edit = QLineEdit()
        self.label_path_edit.setPlaceholderText("标注文件夹路径")
        browse_image_btn = QPushButton("浏览图片")
        browse_image_btn.clicked.connect(self.browse_image_dir)
        browse_label_btn = QPushButton("浏览标注")
        browse_label_btn.clicked.connect(self.browse_label_dir)
        path_layout.addWidget(self.image_path_edit)
        path_layout.addWidget(browse_image_btn)
        path_layout.addWidget(self.label_path_edit)
        path_layout.addWidget(browse_label_btn)
        layout.addLayout(path_layout)

        # 图片显示区
        self.grid_layout = QGridLayout()
        layout.addLayout(self.grid_layout)

        # 翻页按钮
        btn_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一页")
        self.prev_btn.clicked.connect(self.prev_page)
        self.next_btn = QPushButton("下一页")
        self.next_btn.clicked.connect(self.next_page)
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def browse_image_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹", "/home/chenkejing/database")
        if path:
            self.image_path_edit.setText(path)
            self.image_dir = path
            self.update_file_list()

    def browse_label_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择标注文件夹", "/home/chenkejing/database")
        if path:
            self.label_path_edit.setText(path)
            self.label_dir = path
            self.update_file_list()

    def update_file_list(self):
        if self.image_dir:
            self.files = sorted([f for f in os.listdir(self.image_dir)
                                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            self.index = 0
            self.show_page()

    def draw_bbox(self, img, label_file):
        """绘制YOLO标注框，并标红异常标注"""
        h, w = img.shape[:2]
        if not os.path.exists(label_file):
            return img

        with open(label_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls, x_c, y_c, bw, bh = map(float, parts[:5])

            # 检查是否越界或无效
            invalid = not (0 <= x_c <= 1 and 0 <= y_c <= 1 and 0 < bw <= 1 and 0 < bh <= 1)

            # 转为像素坐标
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)

            # 坐标裁剪，防越界
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 <= x1 or y2 <= y1:
                invalid = True

            # 异常框画红色，正常框画绿色
            color = (0, 0, 255) if invalid else (0, 255, 0)
            thickness = 20  # 框线粗细
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # 文字大小
            fontScale = max(0.6, (w + h) / 1000)  # 根据图片大小自适应
            fontThickness = 20  # 文字笔画粗细
            text = str(int(cls))

            # 文本位置稍微偏上
            text_y = max(12, y1 - 5)
            cv2.putText(img, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, fontThickness)

        return img

    def show_page(self):
        # 清空旧内容
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        if not self.files:
            return

        batch_files = self.files[self.index:self.index + self.batch_size]
        cols = 5  # 每行显示几张
        for i, file in enumerate(batch_files):
            img_path = os.path.join(self.image_dir, file)
            label_path = os.path.join(self.label_dir, os.path.splitext(file)[0] + ".txt")

            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            img = self.draw_bbox(img, label_path)
            img = cv2.resize(img, (200, 200))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)

            # 图片 QLabel
            img_label = QLabel()
            img_label.setPixmap(pix)
            img_label.setAlignment(Qt.AlignCenter)

            # 文件名 QLabel
            name_label = QLabel(file)
            name_label.setAlignment(Qt.AlignCenter)
            name_label.setStyleSheet("color: gray; font-size: 11px;")

            # 垂直布局：图片 + 文件名
            container = QVBoxLayout()
            frame = QFrame()
            frame.setLayout(container)
            container.addWidget(img_label)
            container.addWidget(name_label)

            self.grid_layout.addWidget(frame, i // cols, i % cols)

    def next_page(self):
        if self.index + self.batch_size < len(self.files):
            self.index += self.batch_size
            self.show_page()

    def prev_page(self):
        if self.index - self.batch_size >= 0:
            self.index -= self.batch_size
            self.show_page()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LabelChecker()
    window.show()
    sys.exit(app.exec_())
