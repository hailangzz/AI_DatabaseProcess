import sys
import os
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFrame, QRadioButton, QButtonGroup
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/home/chenkejing/anaconda3/plugins/platforms"


# ------------------ 可点击图片 ------------------
class ClickableLabel(QLabel):
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

        # 当前任务类型
        self.task_type = "det"   # det | seg

        self.init_ui()

    # ---------------- UI ----------------
    def init_ui(self):
        main_layout = QVBoxLayout()

        # ===== 路径选择 =====
        path_layout = QHBoxLayout()
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setPlaceholderText("图片文件夹路径")
        self.label_path_edit = QLineEdit()
        self.label_path_edit.setPlaceholderText("标注文件夹路径")

        img_btn = QPushButton("浏览图片")
        lab_btn = QPushButton("浏览标注")
        img_btn.clicked.connect(self.browse_image_dir)
        lab_btn.clicked.connect(self.browse_label_dir)

        path_layout.addWidget(self.image_path_edit)
        path_layout.addWidget(img_btn)
        path_layout.addWidget(self.label_path_edit)
        path_layout.addWidget(lab_btn)
        main_layout.addLayout(path_layout)

        # ===== 任务选择 =====
        task_layout = QHBoxLayout()
        task_layout.addWidget(QLabel("任务类型："))

        self.det_radio = QRadioButton("检测 (Detection)")
        self.seg_radio = QRadioButton("分割 (Segmentation)")
        self.det_radio.setChecked(True)

        self.task_group = QButtonGroup()
        self.task_group.addButton(self.det_radio)
        self.task_group.addButton(self.seg_radio)

        self.det_radio.toggled.connect(self.on_task_change)

        task_layout.addWidget(self.det_radio)
        task_layout.addWidget(self.seg_radio)
        task_layout.addStretch()
        main_layout.addLayout(task_layout)

        # ===== 图片区 =====
        self.grid_layout = QGridLayout()
        main_layout.addLayout(self.grid_layout)

        # ===== 翻页 =====
        btn_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一页 (A)")
        self.next_btn = QPushButton("下一页 (D)")
        self.prev_btn.clicked.connect(self.prev_page)
        self.next_btn.clicked.connect(self.next_page)
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

    # ---------------- 任务切换 ----------------
    def on_task_change(self):
        self.task_type = "det" if self.det_radio.isChecked() else "seg"
        self.show_page()

    # ---------------- 文件夹 ----------------
    def browse_image_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹", "/home/chenkejing/database")
        if path:
            self.image_dir = path
            self.image_path_edit.setText(path)
            self.update_file_list()

    def browse_label_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择标注文件夹", "/home/chenkejing/database")
        if path:
            self.label_dir = path
            self.label_path_edit.setText(path)
            self.update_file_list()

    def update_file_list(self):
        if not self.image_dir:
            return
        self.files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])
        self.index = 0
        self.show_page()

    # ---------------- Detection ----------------
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

    # ---------------- Segmentation ----------------
    # def draw_segmentation(self, img, label_file, alpha=0.4):
    #     h, w = img.shape[:2]
    #     if not os.path.exists(label_file):
    #         return img
    #
    #     overlay = img.copy()
    #     with open(label_file) as f:
    #         for line in f:
    #             p = list(map(float, line.strip().split()))
    #             if len(p) < 7:
    #                 continue
    #             cls = int(p[0])
    #             coords = p[1:]
    #
    #             pts = []
    #             for i in range(0, len(coords), 2):
    #                 pts.append([
    #                     int(coords[i] * w),
    #                     int(coords[i + 1] * h)
    #                 ])
    #             pts = np.array(pts, np.int32)
    #
    #             color = (0, 255, 0)
    #             cv2.fillPoly(overlay, [pts], color)
    #             cv2.polylines(img, [pts], True, color, 2)
    #
    #     cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    #     return img

    def draw_segmentation(self, img, label_file, alpha=0.4):
        h, w = img.shape[:2]
        if not os.path.exists(label_file):
            return img

        # 1️⃣ 先合成一个 mask
        mask = np.zeros((h, w), dtype=np.uint8)

        with open(label_file) as f:
            for line in f:
                p = list(map(float, line.strip().split()))
                if len(p) < 7:
                    continue

                coords = p[1:]
                pts = []
                for i in range(0, len(coords), 2):
                    pts.append([
                        int(coords[i] * w),
                        int(coords[i + 1] * h)
                    ])
                pts = np.array(pts, np.int32)

                cv2.fillPoly(mask, [pts], 255)

        # 2️⃣ 用 mask 叠加显示（洞自然是透明的）
        overlay = img.copy()
        overlay[mask == 255] = (0, 255, 0)

        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # 3️⃣ 轮廓线（可选）
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

        return img

    # ---------------- 显示页面 ----------------
    def show_page(self):
        for i in reversed(range(self.grid_layout.count())):
            w = self.grid_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

        if not self.files:
            return

        self.selected_labels.clear()
        batch = self.files[self.index:self.index + self.batch_size]

        for i, fname in enumerate(batch):
            img_path = os.path.join(self.image_dir, fname)
            label_path = os.path.join(
                self.label_dir,
                os.path.splitext(fname)[0] + ".txt"
            )

            img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            if self.task_type == "det":
                img = self.draw_detection(img, label_path)
            else:
                img = self.draw_segmentation(img, label_path)

            img = cv2.resize(img, (200, 200))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(img.data, 200, 200, img.strides[0], QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)

            img_label = ClickableLabel(img_path)
            img_label.setPixmap(pix)
            img_label.setAlignment(Qt.AlignCenter)

            name_label = QLabel(fname)
            name_label.setAlignment(Qt.AlignCenter)
            name_label.setStyleSheet("font-size:11px;color:gray")

            box = QVBoxLayout()
            frame = QFrame()
            frame.setLayout(box)
            box.addWidget(img_label)
            box.addWidget(name_label)

            self.grid_layout.addWidget(frame, i // 5, i % 5)
            self.selected_labels[img_path] = img_label

    # ---------------- 翻页 ----------------
    def next_page(self):
        if self.index + self.batch_size < len(self.files):
            self.index += self.batch_size
            self.show_page()

    def prev_page(self):
        if self.index - self.batch_size >= 0:
            self.index -= self.batch_size
            self.show_page()

    # ---------------- 快捷键 ----------------
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_S):
            to_delete = [p for p, l in self.selected_labels.items() if l.selected]
            for p in to_delete:
                os.remove(p)
                lp = os.path.join(
                    self.label_dir,
                    os.path.splitext(os.path.basename(p))[0] + ".txt"
                )
                if os.path.exists(lp):
                    os.remove(lp)
            self.update_file_list()

        elif event.key() == Qt.Key_A:
            self.prev_page()
        elif event.key() == Qt.Key_D:
            self.next_page()
        else:
            super().keyPressEvent(event)


# ================= main =================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = LabelChecker()
    win.show()
    sys.exit(app.exec_())
