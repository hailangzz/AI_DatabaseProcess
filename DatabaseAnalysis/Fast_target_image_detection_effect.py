"""
高性能图片检查器
- 支持常见图片格式（jpg/jpeg/png/bmp/webp/tif/tiff
- 使用多线程加载图片，提升响应速度
- LRU缓存机制，减少内存占用
- 点击图片可选中，选中状态会有明显标记
- 支持键盘快捷键：A/D翻页，Delete/S删除选中图片  
"""
import os
import sys
from collections import OrderedDict

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QObject, QRunnable, QThreadPool, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFrame,
    QProgressBar,
)
from PyQt5.QtWidgets import QSlider

# 如有需要修改为你的 Qt 插件目录
os.environ[
    "QT_QPA_PLATFORM_PLUGIN_PATH"
] = "/home/chenkejing/anaconda3/plugins/platforms"


class ClickableLabel(QLabel):
    def __init__(self, file_path=""):
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

            painter.drawLine(int(w * 0.2), int(h * 0.5), int(w * 0.45), int(h * 0.75))

            painter.drawLine(int(w * 0.45), int(h * 0.75), int(w * 0.8), int(h * 0.25))

            painter.end()


class WorkerSignals(QObject):
    finished = pyqtSignal(str, object)


class ImageLoader(QRunnable):
    def __init__(self, path, signals):
        super().__init__()
        self.path = path
        self.signals = signals

    def run(self):

        try:
            img = cv2.imdecode(
                np.fromfile(self.path, dtype=np.uint8), cv2.IMREAD_UNCHANGED
            )

            if img is None:
                return

            img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)

            if len(img.shape) == 3 and img.shape[2] == 4:

                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

                qimg = QImage(
                    img.data,
                    img.shape[1],
                    img.shape[0],
                    img.strides[0],
                    QImage.Format_RGBA8888,
                ).copy()

            else:

                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                qimg = QImage(
                    img.data,
                    img.shape[1],
                    img.shape[0],
                    img.strides[0],
                    QImage.Format_RGB888,
                ).copy()

            self.signals.finished.emit(self.path, qimg)

        except Exception as e:
            print(e)


class LRUCache:
    def __init__(self, max_size=500):
        self.max_size = max_size
        self.cache = OrderedDict()

    def get(self, key):

        if key not in self.cache:
            return None

        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):

        self.cache[key] = value
        self.cache.move_to_end(key)

        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)


class ImageChecker(QWidget):
    def __init__(self):

        super().__init__()

        self.setWindowTitle("高性能图片检查器")
        self.resize(1400, 900)

        self.image_dir = ""
        self.files = []
        self.index = 0
        self.batch_size = 10
        self.image_size = 220

        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(8)

        self.cache = LRUCache(500)
        self.loading_set = set()
        self.signals_holder = []

        self.selected_labels = {}

        self.init_ui()

    def update_image_size(self, value):

        self.image_size = value

        for label in self.image_labels:
            label.setFixedSize(value, value)

        self.show_page()

    def init_ui(self):

        layout = QVBoxLayout()

        path_layout = QHBoxLayout()

        self.image_path_edit = QLineEdit()

        browse_btn = QPushButton("浏览图片")
        browse_btn.clicked.connect(self.browse_image_dir)

        path_layout.addWidget(self.image_path_edit)

        path_layout.addWidget(browse_btn)

        layout.addLayout(path_layout)

        progress_layout = QHBoxLayout()

        self.progress_bar = QProgressBar()

        self.size_slider = QSlider(Qt.Horizontal)
        # 滑块范围设置为 120 到 320，初始值为 220
        self.size_slider.setMinimum(220)
        self.size_slider.setMaximum(520)
        self.size_slider.setValue(280)
        self.size_slider.setFixedWidth(220)
        self.size_slider.valueChanged.connect(self.update_image_size)

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(QLabel("图片大小"))
        progress_layout.addWidget(self.size_slider)

        layout.addLayout(progress_layout)

        self.grid_layout = QGridLayout()
        layout.addLayout(self.grid_layout)

        self.image_labels = []
        self.name_labels = []

        cols = 5

        for i in range(10):
            img_label = ClickableLabel()
            img_label.setFixedSize(self.image_size, self.image_size)
            img_label.setScaledContents(True)
            img_label.setAlignment(Qt.AlignCenter)

            name_label = QLabel()
            name_label.setAlignment(Qt.AlignCenter)

            frame = QFrame()
            frame.setMinimumWidth(self.image_size + 20)

            vbox = QVBoxLayout()

            vbox.addWidget(img_label)
            vbox.addWidget(name_label)

            frame.setLayout(vbox)

            self.grid_layout.addWidget(frame, i // cols, i % cols)

            self.image_labels.append(img_label)

            self.name_labels.append(name_label)

        btn_layout = QHBoxLayout()

        self.prev_btn = QPushButton("上一页(A)")
        self.next_btn = QPushButton("下一页(D)")

        self.prev_btn.clicked.connect(self.prev_page)

        self.next_btn.clicked.connect(self.next_page)

        btn_layout.addWidget(self.prev_btn)

        btn_layout.addWidget(self.next_btn)

        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def browse_image_dir(self):

        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")

        if path:
            self.image_dir = path

            self.image_path_edit.setText(path)

            self.update_file_list()

    def update_file_list(self):

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

        files = []

        for entry in os.scandir(self.image_dir):

            if not entry.is_file():
                continue

            ext = os.path.splitext(entry.name)[1].lower()

            if ext in exts:
                files.append(entry.name)

        self.files = sorted(files)

        self.progress_bar.setMaximum(len(self.files))

        self.index = 0

        self.show_page()

    def request_image(self, img_path):

        if img_path in self.loading_set:
            return

        self.loading_set.add(img_path)

        signals = WorkerSignals()

        signals.finished.connect(self.image_loaded)

        self.signals_holder.append(signals)

        task = ImageLoader(img_path, signals)

        self.thread_pool.start(task)

    def image_loaded(self, path, qimg):

        self.loading_set.discard(path)

        self.cache.put(path, qimg)

        batch_files = self.files[self.index:self.index + self.batch_size]

        for i, file in enumerate(batch_files):

            full_path = os.path.join(self.image_dir, file)

            if full_path == path:
                pix = QPixmap.fromImage(qimg)

                pix = pix.scaled(
                    self.image_labels[i].size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )

                self.image_labels[i].setPixmap(pix)

                break

    def preload_next_page(self):

        start = self.index + self.batch_size
        end = start + self.batch_size

        for file in self.files[start:end]:

            img_path = os.path.join(self.image_dir, file)

            if self.cache.get(img_path) is None:
                self.request_image(img_path)

    def show_page(self):

        if not self.files:
            return

        batch_files = self.files[self.index: self.index + self.batch_size]

        self.selected_labels = {}

        for i in range(10):

            if i >= len(batch_files):
                self.image_labels[i].clear()
                self.name_labels[i].setText("")
                continue

            file = batch_files[i]

            img_path = os.path.join(self.image_dir, file)

            self.image_labels[i].clear()
            self.image_labels[i].selected = False
            self.image_labels[i].file_path = img_path

            self.name_labels[i].setText(file)

            self.selected_labels[img_path] = self.image_labels[i]

            qimg = self.cache.get(img_path)

            if qimg is not None:

                self.image_labels[i].setPixmap(QPixmap.fromImage(qimg))

            else:

                self.request_image(img_path)

        self.preload_next_page()

        self.progress_bar.setValue(min(self.index + self.batch_size, len(self.files)))

    def next_page(self):

        if self.index + self.batch_size < len(self.files):
            self.index += self.batch_size
            self.show_page()

    def prev_page(self):

        if self.index >= self.batch_size:
            self.index -= self.batch_size
            self.show_page()

    def delete_selected(self):

        to_delete = [
            label.file_path for label in self.selected_labels.values() if label.selected
        ]

        if not to_delete:
            return

        deleted_names = set()

        for path in to_delete:

            try:

                if os.path.exists(path):
                    os.remove(path)

                txt_path = os.path.splitext(path)[0] + ".txt"

                if os.path.exists(txt_path):
                    os.remove(txt_path)

                deleted_names.add(os.path.basename(path))

            except Exception as e:
                print(e)

        self.files = [f for f in self.files if f not in deleted_names]

        if self.index >= len(self.files):
            self.index = max(0, len(self.files) - self.batch_size)

        self.progress_bar.setMaximum(len(self.files))

        self.show_page()

    def keyPressEvent(self, event):

        if event.key() == Qt.Key_D:
            self.next_page()

        elif event.key() == Qt.Key_A:
            self.prev_page()

        elif event.key() in (Qt.Key_Delete, Qt.Key_S):

            self.delete_selected()

        else:
            super().keyPressEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    win = ImageChecker()
    win.show()

    sys.exit(app.exec_())
