import sys
import os
import cv2
import numpy as np
import shutil

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFrame, QSlider, QDialog
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_QPA_FONTDIR", None)

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
        if event.modifiers() == Qt.ShiftModifier:
            self.parent_checker.shift_select(self)
        else:
            self.selected = not self.selected
            self.parent_checker.last_clicked = self
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
        self.last_clicked = None
        self.img_size = 200
        self.viewer = None
        self.move_history = []

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # 路径选择
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

        # 图片尺寸滑动条
        slider_layout = QHBoxLayout()
        slider_label = QLabel("图片尺寸")
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(100)
        self.size_slider.setMaximum(500)
        self.size_slider.setValue(self.img_size)
        self.size_slider.valueChanged.connect(self.update_image_size)
        slider_layout.addWidget(slider_label)
        slider_layout.addWidget(self.size_slider)
        layout.addLayout(slider_layout)

        # 图片网格
        self.grid_layout = QGridLayout()
        layout.addLayout(self.grid_layout)

        # 翻页按钮
        btn_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一页 (A)")
        self.prev_btn.clicked.connect(self.prev_page)
        self.next_btn = QPushButton("下一页 (D)")
        self.next_btn.clicked.connect(self.next_page)
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    # ---------------- 文件夹选择 ----------------
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

    # ---------------- 更新文件列表 ----------------
    def update_file_list(self, keep_index=True):
        if self.image_dir:
            self.files = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if not keep_index:
                self.index = 0
            else:
                # 保证 index 不超出范围
                max_index = max(len(self.files) - self.batch_size, 0)
                self.index = min(self.index, max_index)
            self.show_page()

    # ---------------- 显示页面 ----------------
    def show_page(self):
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        if not self.files:
            return

        batch_files = self.files[self.index:self.index+self.batch_size]
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

            self.grid_layout.addWidget(frame, i//cols, i%cols)

    # ---------------- Shift 批量选中 ----------------
    def shift_select(self, clicked_label):
        if not self.last_clicked:
            clicked_label.selected = True
            return
        labels = list(self.selected_labels.values())
        i1 = labels.index(self.last_clicked)
        i2 = labels.index(clicked_label)
        start, end = min(i1,i2), max(i1,i2)
        for i in range(start,end+1):
            labels[i].selected = True
            labels[i].update()

    # ---------------- 图片尺寸更新 ----------------
    def update_image_size(self, value):
        self.img_size = value
        self.show_page()

    # ---------------- 翻页 ----------------
    def next_page(self):
        if self.index + self.batch_size < len(self.files):
            self.index += self.batch_size
        self.show_page()

    def prev_page(self):
        if self.index - self.batch_size >= 0:
            self.index -= self.batch_size
        self.show_page()

    # ---------------- 唯一文件路径 ----------------
    def unique_path(self, path):
        base, ext = os.path.splitext(path)
        counter = 1
        while os.path.exists(path):
            path = f"{base}_{counter}{ext}"
            counter += 1
        return path

    # ---------------- 快捷键 ----------------
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S:
            # 移动图片到目标文件夹（无提示框）
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
            # 保留当前页
            self.update_file_list(keep_index=True)

        elif event.key() == Qt.Key_Z:
            # 撤销上一步移动
            if not self.move_history:
                return
            last_move = self.move_history.pop()
            for dst, src in last_move:
                if os.path.exists(dst):
                    shutil.move(dst, src)
            self.update_file_list(keep_index=True)

        elif event.key() == Qt.Key_W:
            # 打开/关闭原图窗口
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