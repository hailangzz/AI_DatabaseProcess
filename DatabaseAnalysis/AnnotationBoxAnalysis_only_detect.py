import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/home/chenkejing/anaconda3/plugins/platforms"


class ClickableLabel(QLabel):
    """自定义 QLabel，可以响应点击事件"""
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.selected = False
        self.overlay_pix = None  # 用于叠加对勾

    def mousePressEvent(self, event):
        self.selected = not self.selected
        self.update()  # 触发重绘

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.selected:
            painter = QPainter(self)
            pen = QPen(QColor(0, 255, 0), 5)
            painter.setPen(pen)
            # 画一个勾
            w, h = self.width(), self.height()
            painter.drawLine(int(w*0.2), int(h*0.5), int(w*0.45), int(h*0.75))
            painter.drawLine(int(w*0.45), int(h*0.75), int(w*0.8), int(h*0.25))
            painter.end()


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
        self.selected_labels = {}  # 保存 ClickableLabel 对象

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
        """绘制YOLO标注框"""
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
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            color = (0, 255, 0)
            thickness = max(1, (w+h)//300)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
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
        cols = 5
        self.selected_labels = {}
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

            # 使用自定义 ClickableLabel
            img_label = ClickableLabel(img_path)
            img_label.setPixmap(pix)
            img_label.setAlignment(Qt.AlignCenter)
            self.selected_labels[img_path] = img_label

            name_label = QLabel(file)
            name_label.setAlignment(Qt.AlignCenter)
            name_label.setStyleSheet("color: gray; font-size: 11px;")

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

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_S):  # Delete 或 S 都触发删除
            # 删除选中图片和对应label
            to_delete = [label.file_path for label in self.selected_labels.values() if label.selected]
            for path in to_delete:
                if os.path.exists(path):
                    os.remove(path)
                label_path = os.path.join(self.label_dir, os.path.splitext(os.path.basename(path))[0] + ".txt")
                if os.path.exists(label_path):
                    os.remove(label_path)
            self.update_file_list()
        elif event.key() == Qt.Key_A:  # 按 A 翻上一页
            self.prev_page()
        elif event.key() == Qt.Key_D:  # 按 D 翻下一页
            self.next_page()
        else:
            super().keyPressEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LabelChecker()
    window.show()
    sys.exit(app.exec_())
