import sys
import os
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFrame, QRadioButton, QButtonGroup, QProgressBar, QSlider
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt


# =========================
# Qt插件路径（必须在 QApplication 之前更安全）
# =========================
def setup_qt_env():
    qt_plugin_path = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")
    if not qt_plugin_path:
        # 你可以改成更通用的方式
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = \
            os.path.join(os.path.dirname(sys.executable), "plugins/platforms")


# ------------------ 可点击图片 ------------------
class ClickableLabel(QLabel):
    def __init__(self, file_path, index, parent):
        super().__init__()
        self.file_path = file_path
        self.index = index
        self.parent_widget = parent
        self.selected = False

    def mousePressEvent(self, event):
        modifiers = QApplication.keyboardModifiers()

        # SHIFT 连续选择
        if modifiers == Qt.ShiftModifier and self.parent_widget.last_clicked_index is not None:

            start = min(self.parent_widget.last_clicked_index, self.index)
            end = max(self.parent_widget.last_clicked_index, self.index)

            for i in range(start, end + 1):
                label = self.parent_widget.current_batch[i]
                label.selected = True
                label.update()

        else:
            self.selected = not self.selected
            self.update()

        self.parent_widget.last_clicked_index = self.index

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.selected:
            painter = QPainter(self)
            pen = QPen(QColor(0, 255, 0), 5)
            painter.setPen(pen)

            w, h = self.width(), self.height()
            painter.drawLine(int(w * 0.2), int(h * 0.5),
                             int(w * 0.45), int(h * 0.75))
            painter.drawLine(int(w * 0.45), int(h * 0.75),
                             int(w * 0.8), int(h * 0.25))
            painter.end()


# ================= 主窗口 =================
class LabelChecker(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLO 标注质量校验工具（Detection / Segmentation）")
        self.resize(1200, 780)

        self.image_dir = ""
        self.label_dir = ""
        self.files = []
        self.index = 0
        self.batch_size = 10
        self.selected_labels = {}

        self.task_type = "det"
        self.img_size = 420

        self.last_clicked_index = None
        self.current_batch = []

        self.init_ui()

    # ---------------- UI ----------------
    def init_ui(self):
        main_layout = QVBoxLayout()

        path_layout = QHBoxLayout()

        self.image_path_edit = QLineEdit()
        self.label_path_edit = QLineEdit()

        img_btn = QPushButton("浏览图片")
        lab_btn = QPushButton("浏览标注")

        img_btn.clicked.connect(self.browse_image_dir)
        lab_btn.clicked.connect(self.browse_label_dir)

        path_layout.addWidget(self.image_path_edit)
        path_layout.addWidget(img_btn)
        path_layout.addWidget(self.label_path_edit)
        path_layout.addWidget(lab_btn)

        main_layout.addLayout(path_layout)

        # task
        task_layout = QHBoxLayout()

        self.det_radio = QRadioButton("检测")
        self.seg_radio = QRadioButton("分割")
        self.det_radio.setChecked(True)

        self.det_radio.toggled.connect(self.on_task_change)

        task_layout.addWidget(self.det_radio)
        task_layout.addWidget(self.seg_radio)

        self.progress_bar = QProgressBar()
        task_layout.addWidget(self.progress_bar)

        main_layout.addLayout(task_layout)

        # grid
        self.grid_layout = QGridLayout()
        main_layout.addLayout(self.grid_layout)

        # buttons
        btn_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一页 (A)")
        self.next_btn = QPushButton("下一页 (D)")
        self.prev_btn.clicked.connect(self.prev_page)
        self.next_btn.clicked.connect(self.next_page)

        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)

        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

    # ---------------- file ----------------
    def browse_image_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if path:
            self.image_dir = path
            self.image_path_edit.setText(path)
            self.update_file_list()

    def browse_label_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择标注目录")
        if path:
            self.label_dir = path
            self.label_path_edit.setText(path)
            self.update_file_list()

    def update_file_list(self, keep_index=False):
        if not self.image_dir:
            return

        self.files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        if not keep_index:
            self.index = 0

        self.show_page()

    # ---------------- draw det ----------------
    def draw_detection(self, img, label_file):
        h, w = img.shape[:2]

        if not os.path.exists(label_file):
            return img

        with open(label_file) as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 5:
                    continue

                _, xc, yc, bw, bh = map(float, p[:5])

                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return img

    # ---------------- seg ----------------
    def draw_segmentation(self, img, label_file):
        h, w = img.shape[:2]

        if not os.path.exists(label_file):
            return img

        mask = np.zeros((h, w), dtype=np.uint8)

        with open(label_file) as f:
            for line in f:
                p = list(map(float, line.strip().split()))
                if len(p) < 7:
                    continue

                coords = p[1:]
                pts = [[int(coords[i] * w), int(coords[i + 1] * h)]
                       for i in range(0, len(coords), 2)]

                pts = np.array(pts, np.int32)
                cv2.fillPoly(mask, [pts], 255)

        img[mask == 255] = (0, 255, 0)
        return img

    # ---------------- show ----------------
    def show_page(self):

        for i in reversed(range(self.grid_layout.count())):
            w = self.grid_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

        if not self.files:
            return

        self.current_batch = []
        batch = self.files[self.index:self.index + self.batch_size]

        for i, fname in enumerate(batch):

            img_path = os.path.join(self.image_dir, fname)
            label_path = os.path.join(self.label_dir, fname.replace(".jpg", ".txt"))

            img = cv2.imread(img_path)
            if img is None:
                continue

            if self.task_type == "det":
                img = self.draw_detection(img, label_path)
            else:
                img = self.draw_segmentation(img, label_path)

            img = cv2.resize(img, (400, 400))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            qimg = QImage(img.data, 400, 400, img.strides[0], QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)

            label = ClickableLabel(img_path, i, self)
            label.setPixmap(pix)

            self.current_batch.append(label)

            self.grid_layout.addWidget(label, i // 5, i % 5)

        self.progress_bar.setValue(int((self.index / max(1, len(self.files))) * 100))

    # ---------------- nav ----------------
    def next_page(self):
        if self.index + self.batch_size < len(self.files):
            self.index += self.batch_size
            self.show_page()

    def prev_page(self):
        if self.index - self.batch_size >= 0:
            self.index -= self.batch_size
            self.show_page()

    def on_task_change(self):
        self.task_type = "det" if self.det_radio.isChecked() else "seg"
        self.show_page()


# ================= main =================
def main():
    setup_qt_env()

    app = QApplication(sys.argv)
    win = LabelChecker()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()