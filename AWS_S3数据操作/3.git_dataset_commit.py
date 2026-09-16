# -*- coding: utf-8 -*-

"""
Git + DVC 数据批次自动提交脚本

作用：
    根据项目名称、数据批次目录，自动完成：

    1. 进入数据仓库
    2. checkout 对应 dataset branch
    3. dvc add 当前数据批次
    4. git add 相关文件
    5. git commit

例如：

    python git_dataset_commit.py \
        --project carpet \
        --batch date_20260915

最终执行：

    cd /data/database/AITotal_SegmentDatabase

    git checkout dataset/carpet

    dvc add carpetDatabaseSegment/date_20260915/

    git add \
        .gitignore \
        carpetDatabaseSegment/date_20260915.dvc \
        carpetDatabaseSegment/dataset_increment_info.json

    git commit -m "project:carpet

    Data batch:date_20260915

    add carpetDatabaseSegment/date_20260915
    "
"""

import argparse
import os
import subprocess
import sys

# ============================================================
# 配置
# ============================================================

REPO_DIR = "/data/database/AITotal_SegmentDatabase"

# project
#     branch
#     dataset directory
#
# 注意：
#   dataset directory 使用 Git 仓库中的相对路径
#
PROJECT_CONFIG = {
    "carpet": {
        "branch": "dataset/carpet",
        "dataset_dir": "carpetDatabaseSegment",
    },

    "wire": {
        "branch": "dataset/wire",
        "dataset_dir": "wireDatabaseSegment",
    },

    "person": {
        "branch": "dataset/person",
        "dataset_dir": "personDatabaseSegment",
    },

    "plasticbag": {
        "branch": "dataset/plasticbag",
        "dataset_dir": "plasticbagDatabaseSegment",
    },

    "false_positive_carpet": {
        "branch": "dataset/false_positive_carpet",
        "dataset_dir": "FalsePositivesCarpetDatabase",
    },

    "finetune_random_sample": {
        "branch": "dataset/finetune_random_sample",
        "dataset_dir": "finetune_random_sample_datebase",
    },
}


# ============================================================
# 工具函数
# ============================================================

def run_command(command, cwd=REPO_DIR):
    """
    执行 shell 命令。

    command:
        list，例如：
        ["git", "checkout", "dataset/carpet"]

    返回：
        subprocess.CompletedProcess
    """

    print()
    print("=" * 80)
    print("执行命令：")
    print(" ".join(command))
    print("=" * 80)

    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        raise RuntimeError(
            f"命令执行失败，returncode={result.returncode}\n"
            f"命令：{' '.join(command)}"
        )

    return result


def check_git_clean():
    """
    检查当前 Git 是否存在未提交修改。

    这里采用比较保守的策略：
    如果存在未提交修改，则直接终止。

    防止：
        在错误 branch 上带着其他修改执行 checkout，
        导致数据或者 Git 状态混乱。
    """

    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "无法获取 Git 状态：\n" + result.stderr
        )

    if result.stdout.strip():
        print()
        print("=" * 80)
        print("检测到 Git 工作区存在未提交修改：")
        print("=" * 80)
        print(result.stdout)

        raise RuntimeError(
            "Git 工作区不是 clean 状态。\n"
            "为了避免不同数据批次相互污染，程序已停止。"
        )


def get_current_branch():
    """
    获取当前 Git branch。
    """

    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=REPO_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "无法获取当前 Git branch：\n" + result.stderr
        )

    return result.stdout.strip()


def check_dataset_exists(dataset_path):
    """
    检查数据批次目录是否存在。
    """

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"数据批次目录不存在：\n{dataset_path}"
        )


def check_dvc_file_exists(dvc_path):
    """
    检查 dvc add 是否成功生成 .dvc 文件。
    """

    if not os.path.isfile(dvc_path):
        raise RuntimeError(
            f"DVC 文件不存在，说明 dvc add 可能没有正确完成：\n"
            f"{dvc_path}"
        )


# ============================================================
# 核心流程
# ============================================================

def commit_dataset_batch(project, batch):
    """
    提交一个数据批次。

    参数：

        project:
            carpet / wire / person / plasticbag ...

        batch:
            date_20260915
    """

    # --------------------------------------------------------
    # 1. 获取项目配置
    # --------------------------------------------------------

    if project not in PROJECT_CONFIG:
        raise ValueError(
            f"未知 project：{project}\n"
            f"当前支持：{', '.join(PROJECT_CONFIG.keys())}"
        )

    config = PROJECT_CONFIG[project]

    branch = config["branch"]
    dataset_dir = config["dataset_dir"]

    batch_dir = os.path.join(
        dataset_dir,
        batch
    )

    dataset_path = os.path.join(
        REPO_DIR,
        batch_dir
    )

    dvc_file = os.path.join(
        REPO_DIR,
        dataset_dir,
        f"{batch}.dvc"
    )

    increment_info_file = os.path.join(
        REPO_DIR,
        dataset_dir,
        "dataset_increment_info.json"
    )

    # --------------------------------------------------------
    # 2. 打印任务信息
    # --------------------------------------------------------

    print()
    print("#" * 80)
    print("# Git + DVC 数据批次提交")
    print("#" * 80)

    print(f"Project       : {project}")
    print(f"Git Branch    : {branch}")
    print(f"Dataset Dir   : {dataset_dir}")
    print(f"Batch         : {batch}")
    print(f"Batch Path    : {batch_dir}")
    print(f"DVC File      : {dvc_file}")
    print(f"Increment Info: {increment_info_file}")

    print("#" * 80)

    # --------------------------------------------------------
    # 3. 检查批次目录
    # --------------------------------------------------------

    print()
    print("[1/6] 检查数据批次目录...")

    check_dataset_exists(dataset_path)

    print(f"数据批次存在：{dataset_path}")

    # --------------------------------------------------------
    # 4. 检查 Git 工作区
    # --------------------------------------------------------

    print()
    print("[2/6] 检查 Git 工作区...")

    check_git_clean()

    print("Git 工作区 clean。")

    # --------------------------------------------------------
    # 5. checkout 对应 branch
    # --------------------------------------------------------

    print()
    print("[3/6] 切换 Git branch...")

    current_branch = get_current_branch()

    print(f"当前 branch：{current_branch}")

    if current_branch != branch:
        run_command(
            ["git", "checkout", branch]
        )

    current_branch = get_current_branch()

    if current_branch != branch:
        raise RuntimeError(
            f"Git branch 切换失败。\n"
            f"期望：{branch}\n"
            f"实际：{current_branch}"
        )

    print(f"当前 Git branch：{current_branch}")

    # --------------------------------------------------------
    # 6. DVC add
    # --------------------------------------------------------

    print()
    print("[4/6] 执行 DVC add...")

    run_command(
        ["dvc", "add", batch_dir]
    )

    # 检查 DVC 文件
    check_dvc_file_exists(dvc_file)

    print(f"DVC 文件生成成功：{dvc_file}")

    # --------------------------------------------------------
    # 7. Git add
    # --------------------------------------------------------

    print()
    print("[5/6] Git add...")

    # increment info 如果存在则加入
    git_add_files = [
        ".gitignore",
        dvc_file,
    ]

    if os.path.isfile(increment_info_file):
        git_add_files.append(increment_info_file)

    # 转换成相对于 REPO_DIR 的路径
    git_add_files_relative = []

    for path in git_add_files:

        if os.path.isabs(path):
            path = os.path.relpath(
                path,
                REPO_DIR
            )

        git_add_files_relative.append(path)

    run_command(
        ["git", "add"] + git_add_files_relative
    )

    # --------------------------------------------------------
    # 8. 显示 staged 文件
    # --------------------------------------------------------

    print()
    print("当前 Git staged 文件：")

    run_command(
        ["git", "status", "--short"]
    )

    # --------------------------------------------------------
    # 9. Git commit
    # --------------------------------------------------------

    print()
    print("[6/6] Git commit...")

    commit_message = (
        f"project:{project}\n\n"
        f"Data batch:{batch}\n\n"
        f"add {batch_dir}"
    )

    run_command(
        [
            "git",
            "commit",
            "-m",
            commit_message,
        ]
    )

    # --------------------------------------------------------
    # 10. 完成
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("数据批次 Git + DVC 提交成功")
    print("=" * 80)

    print(f"Project    : {project}")
    print(f"Branch     : {branch}")
    print(f"Batch      : {batch}")
    print(f"DVC File   : {os.path.relpath(dvc_file, REPO_DIR)}")
    print("=" * 80)


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Git + DVC 数据批次自动提交工具"
    )

    parser.add_argument(
        "--project",
        required=True,
        choices=PROJECT_CONFIG.keys(),
        help="项目名称，例如 carpet / wire / person / plasticbag",
    )

    parser.add_argument(
        "--batch",
        required=True,
        help="数据批次名称，例如 date_20260915",
    )

    args = parser.parse_args()

    try:

        commit_dataset_batch(
            project=args.project,
            batch=args.batch,
        )

    except Exception as e:

        print()
        print("=" * 80)
        print("数据批次提交失败")
        print("=" * 80)
        print(str(e))
        print("=" * 80)

        sys.exit(1)


if __name__ == "__main__":
    main()

"""


### 使用方式

你现在的 carpet 批次：

```bash
python git_dataset_commit.py \
    --project carpet \
    --batch date_20260915
```

脚本会自动完成：

```text
检查 date_20260915
        ↓
检查 Git clean
        ↓
checkout dataset/carpet
        ↓
dvc add carpetDatabaseSegment/date_20260915/
        ↓
检查 date_20260915.dvc
        ↓
git add
        ↓
git commit
```

最终 commit message 就是：

```text
project:carpet

Data batch:date_20260915

add carpetDatabaseSegment/date_20260915
```

---

### 你的其他项目也可以直接使用

例如 wire：

```bash
python git_dataset_commit.py \
    --project wire \
    --batch date_20260915
```

实际执行：

```bash
git checkout dataset/wire

dvc add wireDatabaseSegment/date_20260915/
```

person：

```bash
python git_dataset_commit.py \
    --project person \
    --batch date_20260915
```

plasticbag：

```bash
python git_dataset_commit.py \
    --project plasticbag \
    --batch date_20260915
```

这样以后你的**模型项目批次处理程序**实际上只需要在最后调用：

```python
commit_dataset_batch(
    project="carpet",
    batch="date_20260915",
)
```
"""
