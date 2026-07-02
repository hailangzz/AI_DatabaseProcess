"""
高性能图片检查器（完整版 + Shift连选修复）
"""
import os
import sys
from collections import OrderedDict

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QObject, QRunnable, QThreadPool, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFrame, QProgressBar, QSlider
)

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/home/chenkejing/anaconda3/plugins/platforms"


# =========================
# 可点击Label
# =========================
class ClickableLabel(QLabel):
    def __init__(self, file_path="", global_index=0, parent_checker=None):
        super().__init__()

        self.file_path = file_path
        self.global_index = global_index
        self.parent_checker = parent_checker
        self.selected = False

    def mousePressEvent(self, event):

        modifiers = QApplication.keyboardModifiers()

        # Shift 连选
        if modifiers & Qt.ShiftModifier:
            if self.parent_checker:
                self.parent_checker.select_range(self.global_index)
            return

        # 普通点击
        self.selected = not self.selected

        if self.parent_checker:
            self.parent_checker.update_selection(self.file_path, self.selected)
            self.parent_checker.last_clicked = self.global_index

        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.selected:
            painter = QPainter(self)

            # 抗锯齿（更清晰）
            painter.setRenderHint(QPainter.Antialiasing)

            w, h = self.width(), self.height()

            # =========================
            # 半透明遮罩（可选，但很专业）
            # =========================
            painter.fillRect(0, 0, w, h, QColor(0, 0, 0, 60))

            # =========================
            # √ 颜色和大小
            # =========================
            alpha = 180  # 0~255  # 字体透明度
            pen = QPen(QColor(0, 255, 0, alpha), 12)
            painter.setPen(pen)

            font = QFont()
            font.setPointSize(int(min(w, h) * 0.5))
            font.setBold(True)
            painter.setFont(font)

            # 居中绘制 √
            painter.drawText(self.rect(), Qt.AlignCenter, "√")

            painter.end()


# =========================
# Worker
# =========================
class WorkerSignals(QObject):
    finished = pyqtSignal(str, object)


class ImageLoader(QRunnable):
    def __init__(self, path, signals):
        super().__init__()
        self.path = path
        self.signals = signals

    def run(self):
        try:
            img = cv2.imdecode(np.fromfile(self.path, np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                return

            img = cv2.resize(img, (200, 200))

            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                fmt = QImage.Format_RGB888
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                fmt = QImage.Format_RGBA8888
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                fmt = QImage.Format_RGB888

            qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], fmt).copy()
            self.signals.finished.emit(self.path, qimg)

        except Exception as e:
            print(e)


# =========================
# LRU
# =========================
class LRUCache:
    def __init__(self, max_size=500):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, k):
        if k not in self.cache:
            return None
        self.cache.move_to_end(k)
        return self.cache[k]

    def put(self, k, v):
        self.cache[k] = v
        self.cache.move_to_end(k)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)


# =========================
# 主界面
# =========================
class ImageChecker(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("高性能图片检查器（完整修复版）")
        self.resize(1400, 900)

        self.image_dir = ""
        self.files = []
        self.index = 0
        self.batch_size = 10
        self.image_size = 220

        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(8)

        self.cache = LRUCache(500)

        self.selected_set = set()
        self.last_clicked = None

        self.labels = []
        self.name_labels = []

        self.init_ui()
        self.undo_stack = []  # 用于撤销删除

    # =========================
    # UI
    # =========================
    def init_ui(self):

        layout = QVBoxLayout()

        # 顶部路径
        top = QHBoxLayout()
        self.path_edit = QLineEdit()
        btn = QPushButton("选择文件夹")
        btn.clicked.connect(self.open_dir)

        top.addWidget(self.path_edit)
        top.addWidget(btn)
        layout.addLayout(top)

        # 进度 + slider
        mid = QHBoxLayout()

        self.progress = QProgressBar()

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(120)
        self.slider.setMaximum(520)
        self.slider.setValue(self.image_size)
        self.slider.valueChanged.connect(self.on_resize)

        mid.addWidget(self.progress)
        mid.addWidget(QLabel("大小"))
        mid.addWidget(self.slider)

        layout.addLayout(mid)

        # grid
        self.grid = QGridLayout()
        layout.addLayout(self.grid)

        cols = 5
        for i in range(10):
            label = ClickableLabel(parent_checker=self)
            label.setFixedSize(self.image_size, self.image_size)
            label.setScaledContents(True)

            name = QLabel()
            name.setAlignment(Qt.AlignCenter)

            frame = QFrame()
            v = QVBoxLayout()
            v.addWidget(label)
            v.addWidget(name)
            frame.setLayout(v)

            self.grid.addWidget(frame, i // cols, i % cols)

            self.labels.append(label)
            self.name_labels.append(name)

        # bottom buttons
        bottom = QHBoxLayout()

        self.prev_btn = QPushButton("上一页(A)")
        self.next_btn = QPushButton("下一页(D)")

        self.prev_btn.clicked.connect(self.prev_page)
        self.next_btn.clicked.connect(self.next_page)

        bottom.addWidget(self.prev_btn)
        bottom.addWidget(self.next_btn)

        layout.addLayout(bottom)

        self.setLayout(layout)

    # =========================
    # 文件
    # =========================
    def open_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择目录")
        if not path:
            return

        self.image_dir = path
        self.path_edit.setText(path)

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        self.files = sorted(
            f for f in os.listdir(path)
            if os.path.splitext(f)[1].lower() in exts
        )

        self.index = 0
        self.show_page()

    # =========================
    # selection update
    # =========================
    def update_selection(self, path, selected):
        if selected:
            self.selected_set.add(path)
        else:
            self.selected_set.discard(path)

    # =========================
    # ⭐ Shift 连选核心
    # =========================
    def select_range(self, current_global):

        if self.last_clicked is None:
            self.last_clicked = current_global
            return

        start = min(self.last_clicked, current_global)
        end = max(self.last_clicked, current_global)

        self.current_batch = self.files[self.index:self.index + self.batch_size]

        for i, label in enumerate(self.labels):

            if i >= len(self.current_batch):
                continue

            global_i = self.index + i
            path = os.path.join(self.image_dir, self.current_batch[i])

            if start <= global_i <= end:
                label.selected = True
                self.selected_set.add(path)
            else:
                label.selected = False
                self.selected_set.discard(path)

            label.update()

        self.last_clicked = current_global

    # =========================
    # 页面
    # =========================
    def show_page(self):

        if not self.files:
            return

        self.current_batch = self.files[self.index:self.index + self.batch_size]

        for i, label in enumerate(self.labels):

            if i >= len(self.current_batch):
                label.clear()
                self.name_labels[i].setText("")
                continue

            file = self.current_batch[i]
            path = os.path.join(self.image_dir, file)

            label.file_path = path
            label.global_index = self.index + i

            label.selected = path in self.selected_set
            label.update()

            self.name_labels[i].setText(file)

            cached = self.cache.get(path)
            if cached:
                label.setPixmap(QPixmap.fromImage(cached))
            else:
                self.load_image(path, label)

        self.progress.setMaximum(len(self.files))
        self.progress.setValue(min(self.index + self.batch_size, len(self.files)))

    # =========================
    # load
    # =========================
    def load_image(self, path, label):

        signals = WorkerSignals()
        signals.finished.connect(lambda p, img: self.on_loaded(p, img, label))

        task = ImageLoader(path, signals)
        self.thread_pool.start(task)

    def on_loaded(self, path, img, label):
        self.cache.put(path, img)
        label.setPixmap(QPixmap.fromImage(img))

    # =========================
    # slider
    # =========================
    def on_resize(self, v):
        self.image_size = v
        for l in self.labels:
            l.setFixedSize(v, v)
        self.show_page()

    # =========================
    # page control
    # =========================
    def next_page(self):
        if self.index + self.batch_size < len(self.files):
            self.index += self.batch_size
            self.show_page()

    def prev_page(self):
        if self.index > 0:
            self.index -= self.batch_size
            self.show_page()

    # =========================
    # keyboard
    # =========================
    def keyPressEvent(self, event):

        if event.key() == Qt.Key_D:
            self.next_page()

        elif event.key() == Qt.Key_A:
            self.prev_page()

        elif event.key() in (Qt.Key_Delete, Qt.Key_S):
            self.delete_selected()

        elif event.key() == Qt.Key_Z and (event.modifiers() & Qt.ControlModifier):
            self.undo_delete()

    # =========================
    # delete
    # =========================
    def delete_selected(self):

        if not self.selected_set:
            return

        deleted_batch = []

        for path in list(self.selected_set):
            try:
                if os.path.exists(path):
                    # 先记录（用于撤销）
                    deleted_batch.append(path)

                    os.remove(path)

            except:
                pass

        # ⭐ 保存这次删除操作
        if deleted_batch:
            self.undo_stack.append(deleted_batch)

        # 更新文件列表
        self.files = [
            f for f in self.files
            if os.path.join(self.image_dir, f) not in self.selected_set
        ]

        self.selected_set.clear()
        self.show_page()

    def undo_delete(self):

        if not self.undo_stack:
            return

        last_deleted = self.undo_stack.pop()

        for path in last_deleted:
            try:
                # 这里无法真正恢复文件（除非用回收站）
                # 所以只能重新加入列表刷新界面
                if os.path.exists(path):
                    continue

                # 重新加入文件列表
                filename = os.path.basename(path)

                if filename not in self.files:
                    self.files.append(filename)

            except:
                pass

        # 重新排序
        self.files = sorted(self.files)

        self.show_page()


# =========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = ImageChecker()
    w.show()
    sys.exit(app.exec_())
