# -*- coding: utf-8 -*-
import re
import sys
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QGroupBox,
    QSizePolicy,
)

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def natural_sort_key(text):
    """
    自然排序。

    例如：

    1.jpg
    2.jpg
    10.jpg

    而不是：

    1.jpg
    10.jpg
    2.jpg
    """

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(
            r"(\d+)",
            text
        )
    ]


class ImageViewer(QLabel):
    """
    图片显示控件

    功能：
    1. 图片自动适应窗口
    2. 窗口缩放时自动重新缩放图片
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.original_pixmap = None

        self.setAlignment(
            Qt.AlignCenter
        )

        self.setText(
            "暂无图片"
        )

        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        self.setMinimumSize(
            300,
            300
        )

    def set_image(self, image_path):

        pixmap = QPixmap(
            str(image_path)
        )

        if pixmap.isNull():
            self.original_pixmap = None

            self.setText(
                f"无法读取图片\n{image_path}"
            )

            self.setPixmap(
                QPixmap()
            )

            return

        self.original_pixmap = pixmap

        self.update_image()

    def update_image(self):

        if self.original_pixmap is None:
            return

        # 当前显示区域
        target_size = self.size()

        # 保持原始比例缩放
        scaled_pixmap = self.original_pixmap.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.setPixmap(
            scaled_pixmap
        )

    def resizeEvent(self, event):

        super().resizeEvent(
            event
        )

        self.update_image()


class ModelResultCompareWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        # ==========================================================
        # 基础配置
        # ==========================================================

        self.setWindowTitle(
            "YOLO 模型预测结果对比工具"
        )

        self.resize(
            1600,
            950
        )

        self.setMinimumSize(
            1000,
            700
        )

        # ==========================================================
        # 数据
        # ==========================================================

        self.model1_dir = None
        self.model2_dir = None

        self.image_names = []

        self.current_index = 0

        # ==========================================================
        # 初始化界面
        # ==========================================================

        self.init_ui()

    # ==========================================================
    # UI
    # ==========================================================

    def init_ui(self):

        central_widget = QWidget()

        self.setCentralWidget(
            central_widget
        )

        main_layout = QVBoxLayout(
            central_widget
        )

        main_layout.setContentsMargins(
            10,
            10,
            10,
            10
        )

        main_layout.setSpacing(
            8
        )

        # ======================================================
        # 标题
        # ======================================================

        title_label = QLabel(
            "YOLO 模型预测结果对比工具"
        )

        title_font = QFont()

        title_font.setPointSize(
            20
        )

        title_font.setBold(
            True
        )

        title_label.setFont(
            title_font
        )

        title_label.setAlignment(
            Qt.AlignCenter
        )

        main_layout.addWidget(
            title_label
        )

        # ======================================================
        # 路径区域
        # ======================================================

        path_layout = QHBoxLayout()

        # ------------------------------------------------------
        # 模型一
        # ------------------------------------------------------

        model1_label = QLabel(
            "模型一预测结果："
        )

        self.model1_path_edit = QLineEdit()

        self.model1_path_edit.setPlaceholderText(
            "请选择模型一预测结果目录"
        )

        model1_button = QPushButton(
            "选择目录"
        )

        model1_button.clicked.connect(
            self.select_model1_dir
        )

        path_layout.addWidget(
            model1_label
        )

        path_layout.addWidget(
            self.model1_path_edit,
            1
        )

        path_layout.addWidget(
            model1_button
        )

        # ------------------------------------------------------
        # 模型二
        # ------------------------------------------------------

        model2_label = QLabel(
            "模型二预测结果："
        )

        self.model2_path_edit = QLineEdit()

        self.model2_path_edit.setPlaceholderText(
            "请选择模型二预测结果目录"
        )

        model2_button = QPushButton(
            "选择目录"
        )

        model2_button.clicked.connect(
            self.select_model2_dir
        )

        path_layout.addWidget(
            model2_label
        )

        path_layout.addWidget(
            self.model2_path_edit,
            1
        )

        path_layout.addWidget(
            model2_button
        )

        main_layout.addLayout(
            path_layout
        )

        # ======================================================
        # 加载按钮 + 信息
        # ======================================================

        control_layout = QHBoxLayout()

        self.load_button = QPushButton(
            "加载并开始对比"
        )

        self.load_button.setMinimumHeight(
            36
        )

        self.load_button.clicked.connect(
            self.load_images
        )

        control_layout.addWidget(
            self.load_button
        )

        self.info_label = QLabel(
            "请选择两个模型的预测结果目录"
        )

        control_layout.addWidget(
            self.info_label
        )

        control_layout.addStretch()

        main_layout.addLayout(
            control_layout
        )

        # ======================================================
        # 左右图片区域
        # ======================================================

        splitter = QSplitter(
            Qt.Horizontal
        )

        # ------------------------------------------------------
        # 模型一
        # ------------------------------------------------------

        model1_group = QGroupBox(
            "模型一"
        )

        model1_layout = QVBoxLayout(
            model1_group
        )

        self.model1_name_label = QLabel(
            "暂无图片"
        )

        self.model1_name_label.setAlignment(
            Qt.AlignCenter
        )

        self.model1_name_label.setStyleSheet(
            "font-weight: bold;"
        )

        self.model1_viewer = ImageViewer()

        model1_layout.addWidget(
            self.model1_name_label
        )

        model1_layout.addWidget(
            self.model1_viewer,
            1
        )

        # ------------------------------------------------------
        # 模型二
        # ------------------------------------------------------

        model2_group = QGroupBox(
            "模型二"
        )

        model2_layout = QVBoxLayout(
            model2_group
        )

        self.model2_name_label = QLabel(
            "暂无图片"
        )

        self.model2_name_label.setAlignment(
            Qt.AlignCenter
        )

        self.model2_name_label.setStyleSheet(
            "font-weight: bold;"
        )

        self.model2_viewer = ImageViewer()

        model2_layout.addWidget(
            self.model2_name_label
        )

        model2_layout.addWidget(
            self.model2_viewer,
            1
        )

        # ======================================================
        # 加入 splitter
        # ======================================================

        splitter.addWidget(
            model1_group
        )

        splitter.addWidget(
            model2_group
        )

        splitter.setSizes(
            [800, 800]
        )

        main_layout.addWidget(
            splitter,
            1
        )

        # ======================================================
        # 底部导航
        # ======================================================

        navigation_layout = QHBoxLayout()

        self.previous_button = QPushButton(
            "← 上一张"
        )

        self.previous_button.setMinimumSize(
            120,
            40
        )

        self.previous_button.clicked.connect(
            self.previous_image
        )

        self.position_label = QLabel(
            "0 / 0"
        )

        position_font = QFont()

        position_font.setPointSize(
            11
        )

        position_font.setBold(
            True
        )

        self.position_label.setFont(
            position_font
        )

        self.position_label.setAlignment(
            Qt.AlignCenter
        )

        self.next_button = QPushButton(
            "下一张 →"
        )

        self.next_button.setMinimumSize(
            120,
            40
        )

        self.next_button.clicked.connect(
            self.next_image
        )

        navigation_layout.addWidget(
            self.previous_button
        )

        navigation_layout.addStretch()

        navigation_layout.addWidget(
            self.position_label
        )

        navigation_layout.addStretch()

        navigation_layout.addWidget(
            self.next_button
        )

        main_layout.addLayout(
            navigation_layout
        )

        # 初始状态
        self.update_navigation_buttons()

    # ==========================================================
    # 选择模型一目录
    # ==========================================================

    def select_model1_dir(self):

        directory = QFileDialog.getExistingDirectory(
            self,
            "选择模型一预测结果目录"
        )

        if not directory:
            return

        self.model1_path_edit.setText(
            directory
        )

    # ==========================================================
    # 选择模型二目录
    # ==========================================================

    def select_model2_dir(self):

        directory = QFileDialog.getExistingDirectory(
            self,
            "选择模型二预测结果目录"
        )

        if not directory:
            return

        self.model2_path_edit.setText(
            directory
        )

    # ==========================================================
    # 扫描图片
    # ==========================================================

    def scan_images(self, directory):

        result = {}

        directory = Path(
            directory
        )

        if not directory.exists():
            return result

        for file in directory.iterdir():

            if not file.is_file():
                continue

            if file.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            result[file.name] = file

        return result

    # ==========================================================
    # 加载图片
    # ==========================================================

    def load_images(self):

        model1_dir = (
            self.model1_path_edit
            .text()
            .strip()
        )

        model2_dir = (
            self.model2_path_edit
            .text()
            .strip()
        )

        if not model1_dir:
            QMessageBox.warning(
                self,
                "提示",
                "请选择模型一预测结果目录"
            )

            return

        if not model2_dir:
            QMessageBox.warning(
                self,
                "提示",
                "请选择模型二预测结果目录"
            )

            return

        if not Path(model1_dir).is_dir():
            QMessageBox.critical(
                self,
                "错误",
                "模型一目录不存在"
            )

            return

        if not Path(model2_dir).is_dir():
            QMessageBox.critical(
                self,
                "错误",
                "模型二目录不存在"
            )

            return

        # 保存目录
        self.model1_dir = Path(
            model1_dir
        )

        self.model2_dir = Path(
            model2_dir
        )

        # 扫描图片
        model1_images = self.scan_images(
            self.model1_dir
        )

        model2_images = self.scan_images(
            self.model2_dir
        )

        if not model1_images:
            QMessageBox.warning(
                self,
                "提示",
                "模型一目录中没有找到图片"
            )

            return

        if not model2_images:
            QMessageBox.warning(
                self,
                "提示",
                "模型二目录中没有找到图片"
            )

            return

        # ======================================================
        # 获取同名图片
        # ======================================================

        common_names = (
                set(model1_images.keys())
                &
                set(model2_images.keys())
        )

        self.image_names = sorted(
            common_names,
            key=natural_sort_key
        )

        only_model1 = (
                set(model1_images.keys())
                -
                set(model2_images.keys())
        )

        only_model2 = (
                set(model2_images.keys())
                -
                set(model1_images.keys())
        )

        self.current_index = 0

        # ======================================================
        # 更新统计信息
        # ======================================================

        self.info_label.setText(
            (
                f"模型一：{len(model1_images)} 张    "
                f"模型二：{len(model2_images)} 张    "
                f"共同图片：{len(self.image_names)} 张    "
                f"模型一独有：{len(only_model1)}    "
                f"模型二独有：{len(only_model2)}"
            )
        )

        if not self.image_names:
            QMessageBox.warning(
                self,
                "提示",
                "两个目录中没有找到同名图片"
            )

            self.clear_images()

            return

        self.show_current_image()

    # ==========================================================
    # 显示当前图片
    # ==========================================================

    def show_current_image(self):

        if not self.image_names:
            return

        filename = self.image_names[
            self.current_index
        ]

        model1_path = (
                self.model1_dir
                / filename
        )

        model2_path = (
                self.model2_dir
                / filename
        )

        # 文件名
        self.model1_name_label.setText(
            filename
        )

        self.model2_name_label.setText(
            filename
        )

        # 图片
        self.model1_viewer.set_image(
            model1_path
        )

        self.model2_viewer.set_image(
            model2_path
        )

        # 位置
        self.position_label.setText(
            f"{self.current_index + 1} / "
            f"{len(self.image_names)}"
        )

        self.update_navigation_buttons()

    # ==========================================================
    # 上一张
    # ==========================================================

    def previous_image(self):

        if not self.image_names:
            return

        if self.current_index > 0:
            self.current_index -= 1

            self.show_current_image()

    # ==========================================================
    # 下一张
    # ==========================================================

    def next_image(self):

        if not self.image_names:
            return

        if (
                self.current_index
                <
                len(self.image_names) - 1
        ):
            self.current_index += 1

            self.show_current_image()

    # ==========================================================
    # 第一张
    # ==========================================================

    def first_image(self):

        if not self.image_names:
            return

        self.current_index = 0

        self.show_current_image()

    # ==========================================================
    # 最后一张
    # ==========================================================

    def last_image(self):

        if not self.image_names:
            return

        self.current_index = (
                len(self.image_names) - 1
        )

        self.show_current_image()

    # ==========================================================
    # 更新按钮
    # ==========================================================

    def update_navigation_buttons(self):

        if not self.image_names:
            self.previous_button.setEnabled(
                False
            )

            self.next_button.setEnabled(
                False
            )

            return

        self.previous_button.setEnabled(
            self.current_index > 0
        )

        self.next_button.setEnabled(
            self.current_index
            <
            len(self.image_names) - 1
        )

    # ==========================================================
    # 清空
    # ==========================================================

    def clear_images(self):

        self.image_names = []

        self.current_index = 0

        self.model1_viewer.setPixmap(
            QPixmap()
        )

        self.model1_viewer.setText(
            "暂无图片"
        )

        self.model2_viewer.setPixmap(
            QPixmap()
        )

        self.model2_viewer.setText(
            "暂无图片"
        )

        self.model1_name_label.setText(
            "暂无图片"
        )

        self.model2_name_label.setText(
            "暂无图片"
        )

        self.position_label.setText(
            "0 / 0"
        )

        self.update_navigation_buttons()

    # ==========================================================
    # 键盘操作
    # ==========================================================

    def keyPressEvent(self, event):

        # A：上一张
        if event.key() == Qt.Key_A:

            self.previous_image()

        # D：下一张
        elif event.key() == Qt.Key_D:

            self.next_image()

        # ←：上一张
        elif event.key() == Qt.Key_Left:

            self.previous_image()

        # →：下一张
        elif event.key() == Qt.Key_Right:

            self.next_image()

        # Home：第一张
        elif event.key() == Qt.Key_Home:

            self.first_image()

        # End：最后一张
        elif event.key() == Qt.Key_End:

            self.last_image()

        else:

            super().keyPressEvent(
                event
            )


def main():
    app = QApplication(
        sys.argv
    )

    # ==========================================================
    # 全局字体
    # ==========================================================

    font = QFont(
        "Noto Sans CJK SC"
    )

    font.setPointSize(
        10
    )

    app.setFont(
        font
    )

    window = ModelResultCompareWindow()

    window.show()

    sys.exit(
        app.exec_()
    )


if __name__ == "__main__":
    main()
