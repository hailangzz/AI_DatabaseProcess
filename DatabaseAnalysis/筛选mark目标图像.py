import sys
import os
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/home/chenkejing/anaconda3/plugins/platforms"


class ClickableLabel(QLabel):
    """可点击图片"""
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.selected = False

    def mousePressEvent(self, event):
        self.selected = not self.selected
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.selected:
            painter = QPainter(self)
            pen = QPen(QColor(0, 255, 0), 5)
            painter.setPen(pen)
            w, h = self.width(), self.height()
            painter.drawLine(int(w*0.2), int(h*0.5), int(w*0.45), int(h*0.75))
            painter.drawLine(int(w*0.45), int(h*0.75), int(w*0.8), int(h*0.25))
            painter.end()


class ImageChecker(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图片检查器")
        self.resize(1200, 800)

        self.image_dir = ""
        self.files = []
        self.index = 0
        self.batch_size = 10
        self.selected_labels = {}

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # 路径选择
        path_layout = QHBoxLayout()
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setPlaceholderText("图片文件夹路径")
        browse_btn = QPushButton("浏览图片")
        browse_btn.clicked.connect(self.browse_image_dir)
        path_layout.addWidget(self.image_path_edit)
        path_layout.addWidget(browse_btn)
        layout.addLayout(path_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        layout.addWidget(self.progress_bar)

        # 图片显示区域
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

    def update_file_list(self):
        if not self.image_dir:
            return
        self.files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'))
        ])
        # 更新进度条总数
        self.progress_bar.setMaximum(len(self.files))
        # 保持当前页 index 合理
        if self.index >= len(self.files):
            self.index = max(0, len(self.files) - self.batch_size)
        self.show_page()

    def read_image(self, img_path):
        """稳定读取各种图片，包括 RGBA PNG"""
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            print("read failed:", img_path)
            return None
        img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)

        # RGBA PNG
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGBA8888).copy()
        else:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888).copy()
        return qimg

    def show_page(self):
        # 清空旧内容
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        if not self.files:
            self.progress_bar.setValue(0)
            return

        batch_files = self.files[self.index:self.index + self.batch_size]
        cols = 5
        self.selected_labels = {}

        for i, file in enumerate(batch_files):
            img_path = os.path.join(self.image_dir, file)
            qimg = self.read_image(img_path)
            if qimg is None:
                continue

            pix = QPixmap.fromImage(qimg)
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

        # 更新进度条
        self.progress_bar.setValue(min(self.index + self.batch_size, len(self.files)))

    def next_page(self):
        if self.index + self.batch_size < len(self.files):
            self.index += self.batch_size
            self.show_page()

    def prev_page(self):
        if self.index - self.batch_size >= 0:
            self.index -= self.batch_size
            self.show_page()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_S):
            to_delete = [label.file_path for label in self.selected_labels.values() if label.selected]
            for path in to_delete:
                if os.path.exists(path):
                    os.remove(path)
                label_path = os.path.splitext(path)[0] + ".txt"
                if os.path.exists(label_path):
                    os.remove(label_path)
                # 删除后从列表中移除
                if os.path.basename(path) in self.files:
                    self.files.remove(os.path.basename(path))
            # 删除后保持当前 index 不变，自动补位
            if self.index >= len(self.files):
                self.index = max(0, len(self.files) - self.batch_size)
            self.show_page()
        elif event.key() == Qt.Key_A:
            self.prev_page()
        elif event.key() == Qt.Key_D:
            self.next_page()
        else:
            super().keyPressEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageChecker()
    window.show()
    sys.exit(app.exec_())