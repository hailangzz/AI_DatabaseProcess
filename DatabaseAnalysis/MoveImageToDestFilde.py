import os
import re  # ✅ 新增
import shutil
import sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFrame, QSlider, QDialog, QProgressBar
)

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_QPA_FONTDIR", None)


# ---------------- 自然排序函数 ----------------
def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


# ---------------- 原图查看窗口 ----------------
class ImageViewer(QDialog):
    """原图查看窗口"""

    def __init__(self, img_path):
        super().__init__()
        self.setWindowTitle("原图查看")
        self.resize(900, 700)

        layout = QVBoxLayout()
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((700, 900, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
        label = QLabel()
        label.setPixmap(QPixmap.fromImage(qimg))
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        self.setLayout(layout)


# ---------------- 可点击图片 ----------------
class ClickableLabel(QLabel):
    def __init__(self, file_path, parent_checker):
        super().__init__()
        self.file_path = file_path
        self.selected = False
        self.parent_checker = parent_checker

    def mousePressEvent(self, event):

        if event.modifiers() & Qt.ShiftModifier:
            self.parent_checker.shift_select(self)

        else:
            self.selected = not self.selected

            # 记录路径，不记录控件
            self.parent_checker.last_clicked_path = self.file_path

        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.selected:
            painter = QPainter(self)
            pen = QPen(QColor(0, 255, 0), 5)
            painter.setPen(pen)
            w, h = self.width(), self.height()
            painter.drawLine(int(w * 0.2), int(h * 0.5), int(w * 0.45), int(h * 0.75))
            painter.drawLine(int(w * 0.45), int(h * 0.75), int(w * 0.8), int(h * 0.25))
            painter.end()


# ---------------- 主窗口 ----------------
class LabelChecker(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图片筛选工具")
        self.resize(1200, 800)

        self.image_dir = ""
        self.target_dir = ""
        self.files = []
        self.batch_size = 10
        self.selected_labels = {}
        self.last_clicked_path = None
        self.img_size = 200
        self.viewer = None
        self.move_history = []

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        path_layout = QHBoxLayout()
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setPlaceholderText("图片文件夹路径")
        self.target_path_edit = QLineEdit()
        self.target_path_edit.setPlaceholderText("目标文件夹路径")
        browse_image_btn = QPushButton("浏览图片")
        browse_image_btn.clicked.connect(self.browse_image_dir)
        browse_target_btn = QPushButton("浏览目标")
        browse_target_btn.clicked.connect(self.browse_target_dir)
        path_layout.addWidget(self.image_path_edit)
        path_layout.addWidget(browse_image_btn)
        path_layout.addWidget(self.target_path_edit)
        path_layout.addWidget(browse_target_btn)
        layout.addLayout(path_layout)

        slider_layout = QHBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        slider_layout.addWidget(self.progress_bar, 1)

        slider_inner_layout = QHBoxLayout()
        slider_label = QLabel("图片尺寸")
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(100)
        self.size_slider.setMaximum(500)
        self.size_slider.setValue(self.img_size)
        self.size_slider.valueChanged.connect(self.update_image_size)
        slider_inner_layout.addWidget(slider_label)
        slider_inner_layout.addWidget(self.size_slider)
        slider_layout.addLayout(slider_inner_layout, 1)

        layout.addLayout(slider_layout)

        self.grid_layout = QGridLayout()
        layout.addLayout(self.grid_layout)

        btn_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一页 (A)")
        self.prev_btn.clicked.connect(self.prev_page)
        self.next_btn = QPushButton("下一页 (D)")
        self.next_btn.clicked.connect(self.next_page)
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def browse_image_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹", "/home/chenkejing/database")
        if path:
            self.image_dir = path
            self.image_path_edit.setText(path)
            self.update_file_list(keep_index=False)

    def browse_target_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择目标文件夹", "/home/chenkejing/database")
        if path:
            self.target_dir = path
            self.target_path_edit.setText(path)

    # ---------------- 修改点在这里 ----------------
    def update_file_list(self, keep_index=True):
        if self.image_dir:
            self.files = sorted(
                [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))],
                key=natural_key
            )
            if not keep_index:
                self.index = 0
            else:
                max_index = max(len(self.files) - self.batch_size, 0)
                self.index = min(self.index, max_index)
            self.show_page()

    def show_page(self):
        self.last_clicked_path = None
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        if not self.files:
            self.progress_bar.setValue(0)
            return

        batch_files = self.files[self.index:self.index + self.batch_size]
        cols = 5
        self.selected_labels.clear()

        for i, file in enumerate(batch_files):
            img_path = os.path.join(self.image_dir, file)
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)

            img_label = ClickableLabel(img_path, self)
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

        total = len(self.files)
        processed = min(self.index + self.batch_size, total)
        percent = int(processed / total * 100)
        self.progress_bar.setValue(percent)

    def shift_select(self, clicked_label):

        labels = list(self.selected_labels.values())

        # 第一次点击
        if self.last_clicked_path is None:
            clicked_label.selected = True
            clicked_label.update()

            self.last_clicked_path = clicked_label.file_path
            return

        # 构建路径索引
        path_to_index = {
            label.file_path: idx
            for idx, label in enumerate(labels)
        }

        # 上一次点击的图片不在当前页
        if self.last_clicked_path not in path_to_index:
            clicked_label.selected = True
            clicked_label.update()

            self.last_clicked_path = clicked_label.file_path
            return

        start_idx = path_to_index[self.last_clicked_path]
        end_idx = path_to_index[clicked_label.file_path]

        start = min(start_idx, end_idx)
        end = max(start_idx, end_idx)

        for i in range(start, end + 1):
            labels[i].selected = True
            labels[i].update()

        self.last_clicked_path = clicked_label.file_path

    def update_image_size(self, value):
        self.img_size = value
        self.show_page()

    def next_page(self):
        if self.index + self.batch_size < len(self.files):
            self.index += self.batch_size
        self.show_page()

    def prev_page(self):
        if self.index - self.batch_size >= 0:
            self.index -= self.batch_size
        self.show_page()

    def unique_path(self, path):
        base, ext = os.path.splitext(path)
        counter = 1
        while os.path.exists(path):
            path = f"{base}_{counter}{ext}"
            counter += 1
        return path

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S:
            if not self.target_dir:
                return
            to_move = [label.file_path for label in self.selected_labels.values() if label.selected]
            if not to_move:
                return
            move_record = []
            for src in to_move:
                filename = os.path.basename(src)
                dst = os.path.join(self.target_dir, filename)
                dst = self.unique_path(dst)
                shutil.move(src, dst)
                move_record.append((dst, src))
            if move_record:
                self.move_history.append(move_record)
            self.update_file_list(keep_index=True)

        elif event.key() == Qt.Key_Z:
            if not self.move_history:
                return
            last_move = self.move_history.pop()
            for dst, src in last_move:
                if os.path.exists(dst):
                    shutil.move(dst, src)
            self.update_file_list(keep_index=True)

        elif event.key() == Qt.Key_W:
            if self.viewer and self.viewer.isVisible():
                self.viewer.close()
                self.viewer = None
                return
            for label in self.selected_labels.values():
                if label.selected:
                    self.viewer = ImageViewer(label.file_path)
                    self.viewer.show()
                    break

        elif event.key() == Qt.Key_A:
            self.prev_page()
        elif event.key() == Qt.Key_D:
            self.next_page()
        else:
            super().keyPressEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LabelChecker()
    window.show()
    sys.exit(app.exec_())
