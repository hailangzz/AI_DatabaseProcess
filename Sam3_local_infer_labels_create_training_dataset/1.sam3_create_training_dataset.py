#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAM3 自动标注结果 -> 指定目标训练数据集

功能：

1. 读取 INFO_OUTPUT_FILE
2. 解析每行：
       Label路径:Prompt1,Prompt2,...

3. 根据 Label 路径匹配 TOTAL_IMAGES_INFO 中的原始图片路径

4. 根据指定 TARGET_PROMPTS 筛选目标

5. 读取 SAM3 YOLO Segmentation Label

6. 只保留 TARGET_PROMPTS 对应 class_id 的 Label

7. 将：
       原始图片
       筛选后的 Label

   拷贝到指定训练数据集：

       TARGET_DATASET/
       ├── images/
       └── labels/

8. 保证：
       image.xxx
       image.txt

   一一对应

9. 如果图片中存在多个目标类别：
       只保留 TARGET_PROMPTS 指定的类别

10. 支持：
       person
       rug or carpet
       Cables or wires on the ground
       Plastic sheets or plastic bags on the ground
       Liquid stains on the ground

11. 异常图片 / Label / 路径不会中断整个任务

12. 最后输出统计信息
"""

# ============================================================
# Imports
# ============================================================

import shutil
from pathlib import Path


# ============================================================
# 创建输出目录
# ============================================================


def create_output_dirs():
    """
    创建：

        TARGET_DATASET/
        ├── images/
        └── labels/
    """

    images_dir = TARGET_DATASET / "images"
    labels_dir = TARGET_DATASET / "labels"

    images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return images_dir, labels_dir


# ============================================================
# 读取 TOTAL_IMAGES_INFO
# ============================================================


def load_image_paths():
    """
    读取 TOTAL_IMAGES_INFO。

    返回：

        {
            图片路径字符串: Path,
            ...
        }

    使用字典是为了后面快速根据图片路径查找。
    """

    if not TOTAL_IMAGES_INFO.exists():
        raise FileNotFoundError(
            f"TOTAL_IMAGES_INFO 不存在："
            f"{TOTAL_IMAGES_INFO}"
        )

    image_paths = {}

    with TOTAL_IMAGES_INFO.open(
            "r",
            encoding="utf-8",
    ) as f:

        for line in f:

            image_path_str = line.strip()

            if not image_path_str:
                continue

            image_path = Path(image_path_str)

            # 保留原始字符串作为 key
            image_paths[image_path_str] = image_path

    print(
        f"[INFO] TOTAL_IMAGES_INFO "
        f"读取图片路径：{len(image_paths)}"
    )

    return image_paths


# ============================================================
# 根据 Label 路径匹配原始图片
# ============================================================


def build_label_to_image_mapping(
        image_paths,
):
    """
    建立：

        Label Path
            ↓
        Original Image Path

    例如：

        原图：
        /xxx/images/abc.jpg

        SAM3 Label：
        /xxx/Sam3_auto_labels/abc.txt

    建立：

        /xxx/Sam3_auto_labels/abc.txt
            ->
        /xxx/images/abc.jpg

    注意：

        这里使用 image.stem 匹配 Label。
    """

    mapping = {}

    for image_path in image_paths.values():
        label_path = LABEL_OUTPUT_DIR / (
            f"{image_path.stem}.txt"
        )

        mapping[str(label_path)] = image_path

    print(
        f"[INFO] 建立 Label -> Image 映射："
        f"{len(mapping)} 条"
    )

    return mapping


# ============================================================
# 读取 INFO_OUTPUT_FILE
# ============================================================


def parse_info_file():
    """
    解析 INFO_OUTPUT_FILE。

    格式：

        /path/xxx.txt:person,rug or carpet

    返回：

        [
            {
                "label_path": Path(...),
                "prompts": [...]
            },
            ...
        ]
    """

    if not INFO_OUTPUT_FILE.exists():
        raise FileNotFoundError(
            f"INFO_OUTPUT_FILE 不存在："
            f"{INFO_OUTPUT_FILE}"
        )

    records = []

    with INFO_OUTPUT_FILE.open(
            "r",
            encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
                f,
                start=1,
        ):

            line = line.strip()

            if not line:
                continue

            # ------------------------------------------------
            # 分割：
            #
            # LabelPath:Prompt1,Prompt2
            #
            # 只分割第一个 :
            # ------------------------------------------------

            if ":" not in line:
                print(
                    f"[WARNING] INFO 第 "
                    f"{line_number} 行格式异常："
                    f"{line}"
                )

                continue

            label_path_str, prompt_string = (
                line.split(":", 1)
            )

            label_path_str = label_path_str.strip()

            prompt_string = prompt_string.strip()

            if not label_path_str:
                print(
                    f"[WARNING] INFO 第 "
                    f"{line_number} 行 Label 路径为空"
                )

                continue

            prompts = []

            if prompt_string:

                for prompt in prompt_string.split(","):

                    prompt = prompt.strip()

                    if prompt and prompt not in prompts:
                        prompts.append(prompt)

            records.append(
                {
                    "label_path": Path(label_path_str),
                    "prompts": prompts,
                    "line_number": line_number,
                }
            )

    print(
        f"[INFO] INFO_OUTPUT_FILE "
        f"解析完成：{len(records)} 条"
    )

    return records


# ============================================================
# 判断 INFO 记录是否包含目标 Prompt
# ============================================================


def get_target_class_ids(
        prompts,
):
    """
    从当前图片的 Prompt 中获取
    TARGET_PROMPTS 对应的 class_id。

    例如：

        prompts:
            [
                "person",
                "rug or carpet",
            ]

        TARGET_PROMPTS:
            {
                "rug or carpet"
            }

        返回：

            {1}
    """

    class_ids = set()

    for prompt in prompts:

        if prompt not in TARGET_PROMPTS:
            continue

        if prompt not in PROMPT_TO_CLASS_ID:
            print(
                f"[WARNING] Prompt 不存在于 "
                f"TEXT_PROMPTS：{prompt}"
            )

            continue

        class_id = PROMPT_TO_CLASS_ID[prompt]

        class_ids.add(class_id)

    return class_ids


# ============================================================
# 读取并筛选 YOLO Label
# ============================================================


def filter_label_lines(
        label_path,
        target_class_ids,
):
    """
    读取 YOLO Segmentation Label。

    原始：

        0 x1 y1 x2 y2 ...
        1 x1 y1 x2 y2 ...
        2 x1 y1 x2 y2 ...

    如果：

        target_class_ids = {1}

    则只保留：

        1 x1 y1 x2 y2 ...

    返回：

        filtered_lines
    """

    if not label_path.exists():
        print(
            f"[WARNING] Label 不存在："
            f"{label_path}"
        )

        return None

    filtered_lines = []

    try:

        with label_path.open(
                "r",
                encoding="utf-8",
        ) as f:

            for line_number, line in enumerate(
                    f,
                    start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                # ------------------------------------------------
                # YOLO Segmentation 最少：
                #
                # class_id
                # x1 y1
                # x2 y2
                # x3 y3
                #
                # 至少需要：
                #
                # 1 + 6 = 7
                # ------------------------------------------------

                if len(parts) < 7:
                    print(
                        f"[WARNING] Label 格式异常："
                        f"{label_path} "
                        f"第 {line_number} 行"
                    )

                    continue

                try:

                    class_id = int(parts[0])

                except ValueError:

                    print(
                        f"[WARNING] class_id 无法解析："
                        f"{label_path} "
                        f"第 {line_number} 行"
                    )

                    continue

                # ------------------------------------------------
                # 筛选 class
                # ------------------------------------------------

                if class_id not in target_class_ids:
                    continue

                filtered_lines.append(line)

    except Exception as e:

        print(
            f"[ERROR] 读取 Label 失败："
            f"{label_path}"
        )

        print(
            f"[ERROR] {type(e).__name__}: {e}"
        )

        return None

    return filtered_lines


# ============================================================
# 复制图片
# ============================================================


def copy_image(
        image_path,
        output_image_path,
):
    """
    复制原始图片。
    """

    if not image_path.exists():
        print(
            f"[WARNING] 原始图片不存在："
            f"{image_path}"
        )

        return False

    if output_image_path.exists():

        if not OVERWRITE_EXISTING:
            print(
                f"[SKIP] 图片已存在："
                f"{output_image_path}"
            )

            return True

    try:

        shutil.copy2(
            image_path,
            output_image_path,
        )

        return True

    except Exception as e:

        print(
            f"[ERROR] 图片复制失败："
            f"{image_path}"
        )

        print(
            f"[ERROR] {type(e).__name__}: {e}"
        )

        return False


# ============================================================
# 保存 Label
# ============================================================


def save_label(
        filtered_lines,
        output_label_path,
):
    """
    保存筛选后的 YOLO Label。
    """

    try:

        if output_label_path.exists():

            if not OVERWRITE_EXISTING:
                print(
                    f"[SKIP] Label 已存在："
                    f"{output_label_path}"
                )

                return True

        with output_label_path.open(
                "w",
                encoding="utf-8",
        ) as f:

            f.write(
                "\n".join(filtered_lines)
            )

            f.write("\n")

        return True

    except Exception as e:

        print(
            f"[ERROR] Label 保存失败："
            f"{output_label_path}"
        )

        print(
            f"[ERROR] {type(e).__name__}: {e}"
        )

        return False


# ============================================================
# 处理单条 INFO 记录
# ============================================================


def process_one_record(
        record,
        label_to_image,
        output_images_dir,
        output_labels_dir,
):
    """
    处理一条 INFO 记录。

    返回：

        success
        skip
        no_target
        failed
    """

    label_path = record["label_path"]

    prompts = record["prompts"]

    # --------------------------------------------------------
    # 判断目标 class
    # --------------------------------------------------------

    target_class_ids = get_target_class_ids(
        prompts
    )

    if not target_class_ids:
        return "skip"

    # --------------------------------------------------------
    # 找原始图片
    # --------------------------------------------------------

    image_path = label_to_image.get(
        str(label_path)
    )

    # --------------------------------------------------------
    # 如果路径字符串因为格式差异没有匹配
    #
    # 再尝试直接根据 stem 搜索
    # --------------------------------------------------------

    if image_path is None:
        print(
            f"[WARNING] 找不到 Label 对应原始图片："
            f"{label_path}"
        )

        return "failed"

    # --------------------------------------------------------
    # Label 必须存在
    # --------------------------------------------------------

    filtered_lines = filter_label_lines(
        label_path=label_path,
        target_class_ids=target_class_ids,
    )

    if filtered_lines is None:
        return "failed"

    # --------------------------------------------------------
    # 理论上 INFO 已经说明有目标
    #
    # 但再次根据 Label 内容验证。
    # --------------------------------------------------------

    if not filtered_lines:
        print(
            f"[WARNING] INFO 中包含目标 Prompt，"
            f"但 Label 中没有对应 class："
            f"{label_path}"
        )

        return "no_target"

    # --------------------------------------------------------
    # 输出路径
    # --------------------------------------------------------

    output_image_path = (
            output_images_dir
            / image_path.name
    )

    output_label_path = (
            output_labels_dir
            / f"{image_path.stem}.txt"
    )

    # --------------------------------------------------------
    # 复制图片
    # --------------------------------------------------------

    image_success = copy_image(
        image_path=image_path,
        output_image_path=output_image_path,
    )

    if not image_success:
        return "failed"

    # --------------------------------------------------------
    # 保存 Label
    # --------------------------------------------------------

    label_success = save_label(
        filtered_lines=filtered_lines,
        output_label_path=output_label_path,
    )

    if not label_success:

        # ----------------------------------------------------
        # 如果图片已经复制成功，但 Label 保存失败，
        # 为了保证 image / label 一一对应，
        # 删除刚刚复制的图片。
        # ----------------------------------------------------

        if (
                output_image_path.exists()
                and not OVERWRITE_EXISTING
        ):

            try:

                output_image_path.unlink()

            except Exception:
                pass

        return "failed"

    # --------------------------------------------------------
    # 成功
    # --------------------------------------------------------

    print(
        f"[SUCCESS] "
        f"{image_path.name}"
    )

    print(
        f"          Prompt: {prompts}"
    )

    print(
        f"          Target class: "
        f"{sorted(target_class_ids)}"
    )

    print(
        f"          Image: "
        f"{output_image_path}"
    )

    print(
        f"          Label: "
        f"{output_label_path}"
    )

    print(
        f"          Objects: "
        f"{len(filtered_lines)}"
    )

    return "success"


# ============================================================
# 主函数
# ============================================================


def main():
    print("\n")

    print("=" * 80)

    print(
        "SAM3 自动标注结果 -> "
        "指定目标训练数据集"
    )

    print("=" * 80)

    print(
        f"[INFO] TOTAL_IMAGES_INFO："
        f"{TOTAL_IMAGES_INFO}"
    )

    print(
        f"[INFO] INFO_OUTPUT_FILE："
        f"{INFO_OUTPUT_FILE}"
    )

    print(
        f"[INFO] LABEL_OUTPUT_DIR："
        f"{LABEL_OUTPUT_DIR}"
    )

    print(
        f"[INFO] TARGET_PROMPTS："
        f"{list(TARGET_PROMPTS)}"
    )

    print(
        f"[INFO] TARGET_DATASET："
        f"{TARGET_DATASET}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # 检查 TARGET_PROMPTS
    # --------------------------------------------------------

    invalid_prompts = (
            TARGET_PROMPTS
            - set(TEXT_PROMPTS)
    )

    if invalid_prompts:
        raise ValueError(
            "以下 TARGET_PROMPTS "
            "不存在于 TEXT_PROMPTS："
            f"{invalid_prompts}"
        )

    # --------------------------------------------------------
    # 创建输出目录
    # --------------------------------------------------------

    (
        output_images_dir,
        output_labels_dir,
    ) = create_output_dirs()

    # --------------------------------------------------------
    # 读取原始图片
    # --------------------------------------------------------

    image_paths = load_image_paths()

    # --------------------------------------------------------
    # 建立：
    #
    # Label -> Image
    #
    # 映射
    # --------------------------------------------------------

    label_to_image = (
        build_label_to_image_mapping(
            image_paths=image_paths
        )
    )

    # --------------------------------------------------------
    # 读取 INFO
    # --------------------------------------------------------

    records = parse_info_file()

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    total_count = len(records)

    success_count = 0
    skip_count = 0
    no_target_count = 0
    failed_count = 0

    # --------------------------------------------------------
    # 开始处理
    # --------------------------------------------------------

    for index, record in enumerate(
            records,
            start=1,
    ):

        print("\n")

        print("-" * 80)

        print(
            f"[{index}/{total_count}] "
            f"处理"
        )

        status = process_one_record(
            record=record,
            label_to_image=label_to_image,
            output_images_dir=output_images_dir,
            output_labels_dir=output_labels_dir,
        )

        # ----------------------------------------------------
        # 统计
        # ----------------------------------------------------

        if status == "success":

            success_count += 1

        elif status == "skip":

            skip_count += 1

        elif status == "no_target":

            no_target_count += 1

        elif status == "failed":

            failed_count += 1

        processed_count = (
                success_count
                + skip_count
                + no_target_count
                + failed_count
        )

        print(
            f"[PROGRESS] "
            f"{processed_count}/{total_count} | "
            f"success={success_count}, "
            f"skip={skip_count}, "
            f"no_target={no_target_count}, "
            f"failed={failed_count}"
        )

    # ========================================================
    # 最终统计
    # ========================================================

    print("\n")

    print("=" * 80)

    print(
        "训练数据集生成完成"
    )

    print("=" * 80)

    print(
        f"[RESULT] INFO 总记录："
        f"{total_count}"
    )

    print(
        f"[RESULT] 成功生成："
        f"{success_count}"
    )

    print(
        f"[RESULT] 非目标跳过："
        f"{skip_count}"
    )

    print(
        f"[RESULT] 无有效目标："
        f"{no_target_count}"
    )

    print(
        f"[RESULT] 处理失败："
        f"{failed_count}"
    )

    print("-" * 80)

    print(
        f"[RESULT] Images："
        f"{output_images_dir}"
    )

    print(
        f"[RESULT] Labels："
        f"{output_labels_dir}"
    )

    print("=" * 80)


# ============================================================
# 配置
# ============================================================

# ------------------------------------------------------------
# 与 SAM3 推理程序保持一致
# ------------------------------------------------------------

TOTAL_IMAGES_INFO = Path(
    "/data/database/aws_origin_sample/images/"
    "total_images_path_info.txt"
)

# ------------------------------------------------------------
# SAM3 Label 输出目录
# ------------------------------------------------------------

LABEL_OUTPUT_DIR = TOTAL_IMAGES_INFO.parent / "Sam3_auto_labels"

# ------------------------------------------------------------
# INFO_OUTPUT_FILE
# ------------------------------------------------------------

TEXT_PROMPTS = [
    "person",
    "rug or carpet",
    "Cables or wires on the ground",
    "Plastic sheets or plastic bags on the ground",
    "Liquid stains on the ground",
]

prompt_string = "_".join(TEXT_PROMPTS)

INFO_OUTPUT_FILE = TOTAL_IMAGES_INFO.parent / (
    f"total_{prompt_string}"
    f"_Sam3_auto_labels_save_info.txt"
)

# ============================================================
# 指定需要生成训练数据集的目标
# ============================================================

TARGET_PROMPTS = {
    # "rug or carpet",
    # "Cables or wires on the ground",
    # "Plastic sheets or plastic bags on the ground",
    # "Liquid stains on the ground",
    "person",
}

# ============================================================
# 目标训练数据集输出目录
# ============================================================

TARGET_DATASET = Path(
    "/data/database/aws_origin_sample/"
    "sam3_person_train_dataset"
)

# ============================================================
# 是否覆盖已经存在的文件
# ============================================================

OVERWRITE_EXISTING = False

# ============================================================
# 是否允许复制同名文件
# ============================================================

# False：
#     如果目标文件已经存在，则跳过
#
# True：
#     覆盖目标文件
#
# 推荐 False
# ============================================================

# ============================================================
# Prompt -> YOLO class_id
# ============================================================

PROMPT_TO_CLASS_ID = {
    prompt: class_id
    for class_id, prompt in enumerate(TEXT_PROMPTS)
}

# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()
