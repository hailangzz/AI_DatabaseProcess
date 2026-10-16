#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO 训练样本集审核工具。Python 3.8+；图像预览需要 Pillow：python -m pip install pillow"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
    ".jfif", ".ppm", ".pgm"
}


class YoloDatasetAuditApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO 训练样本集审核工具")
        self.root.geometry("1100x780")
        self.root.minsize(850, 600)

        self.images_dir = None
        self.labels_dir = None
        self.missing_images = []
        self.current_index = 0
        self.preview_photo = None

        self.images_dir_var = tk.StringVar()
        self.labels_dir_var = tk.StringVar()
        self.status_var = tk.StringVar(value="请选择 images 和 labels 目录，然后点击“扫描缺失标签”。")
        self.progress_var = tk.StringVar(value="尚未扫描")
        self.filename_var = tk.StringVar(value="当前没有待审核图像")
        self._build_ui()

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        paths = ttk.LabelFrame(outer, text="数据集目录", padding=10)
        paths.pack(fill=tk.X)

        ttk.Label(paths, text="Images 目录：").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(paths, textvariable=self.images_dir_var).grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(paths, text="选择目录", command=self.choose_images_dir).grid(row=0, column=2, padx=4, pady=4)

        ttk.Label(paths, text="Labels 目录：").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(paths, textvariable=self.labels_dir_var).grid(row=1, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(paths, text="选择目录", command=self.choose_labels_dir).grid(row=1, column=2, padx=4, pady=4)

        ttk.Button(paths, text="扫描缺失标签", command=self.scan_missing_labels).grid(
            row=0, column=3, rowspan=2, padx=(10, 0), ipadx=8, ipady=5
        )
        paths.columnconfigure(1, weight=1)

        info = ttk.Frame(outer)
        info.pack(fill=tk.X, pady=(10, 6))
        ttk.Label(info, textvariable=self.progress_var, font=("", 11, "bold")).pack(side=tk.LEFT)
        ttk.Label(info, textvariable=self.status_var, foreground="#555").pack(side=tk.RIGHT)

        ttk.Label(outer, textvariable=self.filename_var, justify="left").pack(anchor="w", pady=(0, 6))

        preview = ttk.LabelFrame(outer, text="图像预览", padding=8)
        preview.pack(fill=tk.BOTH, expand=True)
        self.preview_label = ttk.Label(
            preview, text="扫描后将在此处显示缺少标签文件的图像",
            anchor="center", justify="center"
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        buttons = ttk.Frame(outer)
        buttons.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(buttons, text="上一张", command=self.previous_image).pack(side=tk.LEFT, padx=(0, 6), ipadx=8, ipady=4)
        ttk.Button(buttons, text="跳过 / 下一张", command=self.next_image).pack(side=tk.LEFT, padx=6, ipadx=8, ipady=4)
        ttk.Separator(buttons, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)
        ttk.Button(buttons, text="删除当前图像", command=self.delete_current_image).pack(side=tk.LEFT, padx=6, ipadx=8, ipady=4)
        ttk.Button(buttons, text="增加空白标签", command=self.create_empty_label).pack(side=tk.LEFT, padx=6, ipadx=8, ipady=4)
        ttk.Button(buttons, text="重新扫描", command=self.scan_missing_labels).pack(side=tk.RIGHT, padx=6, ipadx=8, ipady=4)

        if not PIL_AVAILABLE:
            self.status_var.set("尚未安装 Pillow；请运行 python -m pip install pillow 以启用图像预览。")

    def choose_images_dir(self):
        directory = filedialog.askdirectory(title="请选择 images 目录")
        if directory:
            self.images_dir_var.set(directory)

    def choose_labels_dir(self):
        directory = filedialog.askdirectory(title="请选择 labels 目录")
        if directory:
            self.labels_dir_var.set(directory)

    def _validate_dirs(self):
        images_text = self.images_dir_var.get().strip()
        labels_text = self.labels_dir_var.get().strip()
        if not images_text or not labels_text:
            messagebox.showwarning("目录未选择", "请先选择 images 和 labels 目录。")
            return False

        images_dir = Path(images_text).expanduser()
        labels_dir = Path(labels_text).expanduser()
        if not images_dir.is_dir():
            messagebox.showerror("目录错误", f"Images 目录不存在：\n{images_dir}")
            return False

        if not labels_dir.is_dir():
            if not messagebox.askyesno("Labels 目录不存在", f"Labels 目录不存在：\n{labels_dir}\n\n是否创建？"):
                return False
            try:
                labels_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                messagebox.showerror("创建失败", f"无法创建 Labels 目录：\n{exc}")
                return False

        try:
            if images_dir.resolve() == labels_dir.resolve():
                messagebox.showerror("目录错误", "Images 和 Labels 不能是同一个目录。")
                return False
        except OSError:
            pass

        self.images_dir = images_dir
        self.labels_dir = labels_dir
        return True

    def scan_missing_labels(self):
        if not self._validate_dirs():
            return

        image_paths = sorted(
            (p for p in self.images_dir.rglob("*")
             if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda p: str(p.relative_to(self.images_dir)).lower()
        )
        self.missing_images = []
        for image_path in image_paths:
            relative_path = image_path.relative_to(self.images_dir)
            label_path = (self.labels_dir / relative_path).with_suffix(".txt")
            if not label_path.is_file():
                self.missing_images.append(image_path)

        self.current_index = 0
        self.status_var.set(f"共扫描 {len(image_paths)} 张图像，缺少标签文件 {len(self.missing_images)} 张。")
        if not self.missing_images:
            self.filename_var.set("审核完成：没有发现缺少标签文件的图像。")
            self.progress_var.set("0 / 0")
            self.preview_label.configure(image="", text="审核完成！所有图像均存在对应的 .txt 文件。")
            self.preview_photo = None
            return
        self.show_current_image()

    def show_current_image(self):
        if not self.missing_images:
            self.filename_var.set("当前没有待审核图像")
            self.progress_var.set("0 / 0")
            self.preview_label.configure(image="", text="没有待审核图像。")
            self.preview_photo = None
            return

        self.current_index = max(0, min(self.current_index, len(self.missing_images) - 1))
        image_path = self.missing_images[self.current_index]
        relative_path = image_path.relative_to(self.images_dir)
        label_path = (self.labels_dir / relative_path).with_suffix(".txt")
        self.filename_var.set(f"图像：{relative_path}\n预期标签：{label_path}")
        self.progress_var.set(f"{self.current_index + 1} / {len(self.missing_images)}（待审核）")

        if not PIL_AVAILABLE:
            self.preview_label.configure(
                image="", text=f"未安装 Pillow，无法预览图像。\n运行：python -m pip install pillow\n\n{image_path}"
            )
            self.preview_photo = None
            return

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                max_width = max(400, self.root.winfo_width() - 100)
                max_height = max(300, self.root.winfo_height() - 260)
                resampling = getattr(Image, "Resampling", Image).LANCZOS
                img.thumbnail((max_width, max_height), resampling)
                photo = ImageTk.PhotoImage(img)
            self.preview_photo = photo
            self.preview_label.configure(image=photo, text="")
            self.status_var.set("当前图像缺少对应的标签文件。")
        except Exception as exc:
            self.preview_photo = None
            self.preview_label.configure(image="", text=f"图像无法预览，但仍可审核文件：\n{image_path}\n\n错误：{exc}")
            self.status_var.set("图像读取失败，请检查文件是否损坏。")

    def previous_image(self):
        if not self.missing_images:
            return
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()
        else:
            self.status_var.set("已经是第一张待审核图像。")

    def next_image(self):
        if not self.missing_images:
            return
        if self.current_index < len(self.missing_images) - 1:
            self.current_index += 1
            self.show_current_image()
        else:
            self.status_var.set("已经是最后一张待审核图像。")

    def _remove_current_from_queue(self):
        self.missing_images.pop(self.current_index)
        if self.current_index >= len(self.missing_images):
            self.current_index = max(0, len(self.missing_images) - 1)
        self.show_current_image()

    def delete_current_image(self):
        if not self.missing_images:
            messagebox.showinfo("没有图像", "当前没有待审核图像。")
            return
        image_path = self.missing_images[self.current_index]
        relative_path = image_path.relative_to(self.images_dir)
        confirmed = messagebox.askyesno(
            "确认删除",
            f"确定要删除这张图像吗？\n\n{relative_path}\n\n"
            "此操作会永久删除图像文件，不会删除其他图像或标签文件。"
        )
        if not confirmed:
            return
        try:
            image_path.unlink()
        except OSError as exc:
            messagebox.showerror("删除失败", f"无法删除图像：\n{image_path}\n\n{exc}")
            return
        self._remove_current_from_queue()
        self.status_var.set(f"已删除图像：{relative_path}")

    def create_empty_label(self):
        if not self.missing_images:
            messagebox.showinfo("没有图像", "当前没有待审核图像。")
            return
        image_path = self.missing_images[self.current_index]
        relative_path = image_path.relative_to(self.images_dir)
        label_path = (self.labels_dir / relative_path).with_suffix(".txt")
        try:
            label_path.parent.mkdir(parents=True, exist_ok=True)
            # exclusive create: 若文件意外已存在，不覆盖它。
            with label_path.open("x", encoding="utf-8"):
                pass
        except FileExistsError:
            messagebox.showwarning("标签已存在", f"标签文件已经存在，不会覆盖：\n{label_path}\n\n将重新扫描。")
            self.scan_missing_labels()
            return
        except OSError as exc:
            messagebox.showerror("创建失败", f"无法创建空白标签文件：\n{label_path}\n\n{exc}")
            return
        self._remove_current_from_queue()
        self.status_var.set(f"已创建空白标签：{label_path}")


def main():
    root = tk.Tk()
    YoloDatasetAuditApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
