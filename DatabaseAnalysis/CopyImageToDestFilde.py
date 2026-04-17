import sys
import os
import cv2
import numpy as np
import shutil

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFrame, QSlider, QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt


os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_QPA_FONTDIR", None)


# ---------------- 图片Label ----------------
class ClickableLabel(QLabel):
    def __init__(self, file_path, parent):
        super().__init__()
        self.file_path = file_path
        self.selected = False
        self.copied = False
        self.parent = parent

    def mousePressEvent(self, event):
        if event.modifiers() == Qt.ShiftModifier:
            self.parent.shift_select(self)
        else:
            self.selected = not self.selected
            self.parent.last_clicked = self
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)

        # 绿色选中
        if self.selected:
            pen = QPen(QColor(0, 255, 0), 5)
            painter.setPen(pen)
            w, h = self.width(), self.height()
            painter.drawLine(int(w*0.2), int(h*0.5), int(w*0.45), int(h*0.75))
            painter.drawLine(int(w*0.45), int(h*0.75), int(w*0.8), int(h*0.25))

        # 蓝色已复制
        if self.copied:
            pen = QPen(QColor(0, 150, 255), 5)
            painter.setPen(pen)
            w, h = self.width(), self.height()
            painter.drawLine(int(w*0.2), int(h*0.3), int(w*0.4), int(h*0.5))
            painter.drawLine(int(w*0.4), int(h*0.5), int(w*0.8), int(h*0.2))

        painter.end()


# ---------------- 主窗口 ----------------
class LabelChecker(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO数据筛选工具")
        self.resize(1200, 800)

        self.image_dir = ""
        self.label_dir = ""
        self.target_dir = ""

        self.files = []
        self.index = 0
        self.batch_size = 10
        self.img_size = 200

        self.selected_labels = {}
        self.last_clicked = None

        self.copied_files = set()
        self.history = []

        self.init_ui()

    # ---------------- UI ----------------
    def init_ui(self):
        layout = QVBoxLayout()

        # 路径
        path_layout = QHBoxLayout()

        self.image_edit = QLineEdit()
        self.image_edit.setPlaceholderText("图片目录")

        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("标签目录")

        self.target_edit = QLineEdit()
        self.target_edit.setPlaceholderText("目标目录")

        btn_img = QPushButton("图片")
        btn_lab = QPushButton("标签")
        btn_tar = QPushButton("目标")

        btn_img.clicked.connect(self.choose_image_dir)
        btn_lab.clicked.connect(self.choose_label_dir)
        btn_tar.clicked.connect(self.choose_target_dir)

        path_layout.addWidget(self.image_edit)
        path_layout.addWidget(btn_img)
        path_layout.addWidget(self.label_edit)
        path_layout.addWidget(btn_lab)
        path_layout.addWidget(self.target_edit)
        path_layout.addWidget(btn_tar)

        layout.addLayout(path_layout)

        # 滑块 + 进度条
        slider_layout = QHBoxLayout()

        self.progress = QProgressBar()
        slider_layout.addWidget(self.progress, 1)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(100)
        self.slider.setMaximum(500)
        self.slider.setValue(self.img_size)
        self.slider.valueChanged.connect(self.change_size)

        slider_layout.addWidget(QLabel("尺寸"))
        slider_layout.addWidget(self.slider, 1)

        layout.addLayout(slider_layout)

        # 图片区
        self.grid = QGridLayout()
        layout.addLayout(self.grid)

        # 翻页按钮
        btn_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一页 (A)")
        self.next_btn = QPushButton("下一页 (D)")
        self.prev_btn.clicked.connect(self.prev_page)
        self.next_btn.clicked.connect(self.next_page)

        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)

        layout.addLayout(btn_layout)

        self.setLayout(layout)

    # ---------------- 选择路径 ----------------
    def choose_image_dir(self):
        path = QFileDialog.getExistingDirectory(self)
        if path:
            self.image_dir = path
            self.image_edit.setText(path)
            self.load_files()

    def choose_label_dir(self):
        path = QFileDialog.getExistingDirectory(self)
        if path:
            self.label_dir = path
            self.label_edit.setText(path)

    def choose_target_dir(self):
        path = QFileDialog.getExistingDirectory(self)
        if path:
            self.target_dir = path
            self.target_edit.setText(path)

    # ---------------- 加载 ----------------
    def load_files(self):
        if not self.image_dir:
            return
        self.files = [f for f in os.listdir(self.image_dir)
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        self.files.sort()  # ⭐ 排序

        self.index = 0
        self.show_page()

    # ---------------- 显示 ----------------
    def show_page(self):
        for i in reversed(range(self.grid.count())):
            w = self.grid.itemAt(i).widget()
            if w:
                w.setParent(None)

        batch = self.files[self.index:self.index+self.batch_size]
        self.selected_labels.clear()

        for i, f in enumerate(batch):
            path = os.path.join(self.image_dir, f)
            img = cv2.imread(path)
            if img is None:
                continue

            img = cv2.resize(img, (self.img_size, self.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            qimg = QImage(img.data, img.shape[1], img.shape[0],
                          img.strides[0], QImage.Format_RGB888)

            label = ClickableLabel(path, self)
            if path in self.copied_files:
                label.copied = True

            label.setPixmap(QPixmap.fromImage(qimg))
            self.selected_labels[path] = label

            v = QVBoxLayout()
            v.addWidget(label)
            v.addWidget(QLabel(f))

            frame = QFrame()
            frame.setLayout(v)

            self.grid.addWidget(frame, i//5, i%5)

        # 进度
        if self.files:
            percent = int(min(self.index + self.batch_size, len(self.files)) / len(self.files) * 100)
            self.progress.setValue(percent)

    # ---------------- 翻页 ----------------
    def next_page(self):
        if self.index + self.batch_size < len(self.files):
            self.index += self.batch_size
            self.show_page()

    def prev_page(self):
        if self.index - self.batch_size >= 0:
            self.index -= self.batch_size
            self.show_page()

    # ---------------- 尺寸 ----------------
    def change_size(self, v):
        self.img_size = v
        self.show_page()

    # ---------------- Shift选择 ----------------
    def shift_select(self, clicked):
        if not self.last_clicked:
            return
        labels = list(self.selected_labels.values())
        i1 = labels.index(self.last_clicked)
        i2 = labels.index(clicked)
        for i in range(min(i1,i2), max(i1,i2)+1):
            labels[i].selected = True
            labels[i].update()

    # ---------------- 唯一路径 ----------------
    def unique(self, path):
        base, ext = os.path.splitext(path)
        i = 1
        while os.path.exists(path):
            path = f"{base}_{i}{ext}"
            i += 1
        return path

    # ---------------- 获取label ----------------
    def get_label(self, img_path):
        if not self.label_dir:
            return None
        name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        path = os.path.join(self.label_dir, name)
        return path if os.path.exists(path) else None

    # ---------------- 快捷键 ----------------
    def keyPressEvent(self, event):

        if event.key() == Qt.Key_S:
            if not self.target_dir:
                print("请选择目标目录")
                return

            record = []
            for lab in self.selected_labels.values():
                if lab.selected:
                    src_img = lab.file_path
                    name = os.path.basename(src_img)

                    dst_img = self.unique(os.path.join(self.target_dir, "images", name))
                    os.makedirs(os.path.dirname(dst_img), exist_ok=True)
                    shutil.copy2(src_img, dst_img)

                    src_label = self.get_label(src_img)
                    if src_label:
                        dst_label = os.path.join(self.target_dir, "labels",
                                                 os.path.basename(src_label))
                        os.makedirs(os.path.dirname(dst_label), exist_ok=True)
                        shutil.copy2(src_label, dst_label)
                    else:
                        dst_label = None

                    self.copied_files.add(src_img)
                    lab.copied = True
                    lab.update()

                    record.append((dst_img, dst_label))

            if record:
                self.history.append(record)

        elif event.key() == Qt.Key_Z:
            if not self.history:
                return

            last = self.history.pop()
            for img, lab in last:
                if os.path.exists(img):
                    os.remove(img)
                if lab and os.path.exists(lab):
                    os.remove(lab)

        elif event.key() == Qt.Key_A:  # ⭐ 上一页
            self.prev_page()

        elif event.key() == Qt.Key_D:  # ⭐ 下一页
            self.next_page()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = LabelChecker()
    w.show()
    sys.exit(app.exec_())


