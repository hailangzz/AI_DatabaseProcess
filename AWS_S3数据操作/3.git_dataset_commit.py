#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Git + DVC 数据批次自动提交脚本

功能：
1. 根据 project 自动选择 Git branch
2. 检查指定数据批次是否存在
3. 执行 dvc add
4. 自动 git add
5. 自动 git commit
6. 不要求整个 Git 仓库必须 clean
7. 允许其他数据项目存在未提交修改
8. 只提交当前 project / batch 相关文件
9. 防止其他文件被意外加入本次 commit
10. 如果需要切换 Git branch，则要求工作区 clean，避免覆盖其他未提交修改

使用示例：

python 3.git_dataset_commit.py \
    --project carpet \
    --batch date_20260928

python 3.git_dataset_commit.py \
    --project wire \
    --batch date_20260928
"""

import argparse
import os
import subprocess
import sys

# =============================================================================
# 配置
# =============================================================================

REPO_DIR = "/data/database/AITotal_SegmentDatabase"

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


# =============================================================================
# Git / DVC 命令执行
# =============================================================================

def run_command(command, cwd=REPO_DIR):
    """
    执行命令，并打印 stdout / stderr。

    如果命令执行失败，则直接抛出 RuntimeError。
    """

    print("\n" + "=" * 80)
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
            print(result.stderr)

        raise RuntimeError(
            "命令执行失败：{}".format(
                " ".join(command)
            )
        )

    return result


# =============================================================================
# Git 状态
# =============================================================================

def get_current_branch():
    """
    获取当前 Git branch。
    """

    result = subprocess.run(
        [
            "git",
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        ],
        cwd=REPO_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "获取当前 Git branch 失败：\n{}".format(
                result.stderr
            )
        )

    return result.stdout.strip()


def get_git_status_porcelain():
    """
    获取 Git status --porcelain 原始结果。
    """

    result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
        ],
        cwd=REPO_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "获取 Git 工作区状态失败：\n{}".format(
                result.stderr
            )
        )

    return result.stdout


def get_git_dirty_files():
    """
    获取 Git 工作区中发生修改的文件。

    返回：
        list[str]

    注意：
        git status --porcelain 的前两个字符是状态码，
        第 3 个字符开始才是路径。
    """

    status = get_git_status_porcelain()

    dirty_files = []

    for line in status.splitlines():

        if not line.strip():
            continue

        # 正常格式：
        #
        # M  xxx
        #  M xxx
        # ?? xxx
        # D  xxx
        #
        # 前两个字符是状态
        path = line[3:]

        # 对 rename 做一个简单处理。
        #
        # 例如：
        # R  old.txt -> new.txt
        #
        # 这里取最终路径。
        if " -> " in path:
            path = path.split(" -> ")[-1]

        dirty_files.append(path)

    return dirty_files


def check_git_clean_for_branch_switch():
    """
    当需要切换 Git branch 时使用。

    由于 checkout 可能覆盖未提交修改，
    所以 branch 切换仍然要求整个工作区 clean。

    注意：
        这里只在需要切换 branch 时调用。
    """

    status = get_git_status_porcelain()

    if status.strip():
        print("\n" + "=" * 80)
        print("无法安全切换 Git Branch")
        print("=" * 80)

        print("当前 Git 工作区存在未提交修改：")
        print()

        print(status)

        print("=" * 80)
        print(
            "为了避免切换数据分支时覆盖其他未提交修改，"
            "程序不会自动 stash、reset 或清理这些修改。"
        )
        print("=" * 80)

        raise RuntimeError(
            "Git 工作区不是 clean 状态，无法安全切换 branch。"
        )


# =============================================================================
# 当前项目相关修改检查
# =============================================================================

def check_current_project_changes(project, batch):
    """
    检查当前 project / batch 相关的 Git 修改。

    与原来的 check_git_clean() 不同：

    不再要求整个仓库 clean。

    允许：
        1. 当前 batch 目录存在修改
        2. 当前 batch.dvc 存在修改
        3. 当前 project 的 dataset_increment_info.json 修改
        4. 当前 project 的旧 project-level DVC 文件删除
        5. .gitignore 修改
        6. 其他 project 存在修改
        7. .idea / README / cvat_dataset 等其他文件存在修改

    但是：
        当前 project 下其他 batch 的修改会给出警告。
    """

    config = PROJECT_CONFIG[project]

    dataset_dir = config["dataset_dir"]

    batch_dir = os.path.join(
        dataset_dir,
        batch,
    )

    batch_dvc = os.path.join(
        dataset_dir,
        "{}.dvc".format(batch),
    )

    increment_info_file = os.path.join(
        dataset_dir,
        "dataset_increment_info.json",
    )

    # 旧的 project-level DVC 文件。
    #
    # 例如：
    #
    # carpetDatabaseSegment.dvc
    #
    # 如果你现在从整个 dataset DVC
    # 迁移到每个 date 一个 DVC 文件，
    # 这个文件可能处于删除状态。
    old_dataset_dvc = "{}.dvc".format(
        dataset_dir
    )

    dirty_files = get_git_dirty_files()

    if not dirty_files:
        print("\nGit 工作区 clean。")

        return

    print("\n" + "=" * 80)
    print("当前 Git 工作区状态")
    print("=" * 80)

    for path in dirty_files:
        print(path)

    print("=" * 80)

    current_project_other_changes = []

    unrelated_changes = []

    for path in dirty_files:

        # ---------------------------------------------------------
        # 当前 batch
        # ---------------------------------------------------------

        if (
                path == batch_dir
                or path.startswith(batch_dir + os.sep)
        ):
            continue

        # ---------------------------------------------------------
        # 当前 batch DVC
        # ---------------------------------------------------------

        if path == batch_dvc:
            continue

        # ---------------------------------------------------------
        # 当前 project 增量信息
        # ---------------------------------------------------------

        if path == increment_info_file:
            continue

        # ---------------------------------------------------------
        # .gitignore
        # ---------------------------------------------------------

        if path == ".gitignore":
            continue

        # ---------------------------------------------------------
        # 当前 project 原来的 project-level DVC
        #
        # 例如：
        #
        # D carpetDatabaseSegment.dvc
        #
        # 如果这是你的预期迁移行为，则允许。
        # ---------------------------------------------------------

        if path == old_dataset_dvc:
            continue

        # ---------------------------------------------------------
        # 当前 project 下其他 batch / 文件
        #
        # 例如：
        #
        # carpetDatabaseSegment/date_20260920/xxx.jpg
        # carpetDatabaseSegment/date_20260921/xxx.jpg
        #
        # 这些不会自动提交，但需要提醒。
        # ---------------------------------------------------------

        if (
                path == dataset_dir
                or path.startswith(dataset_dir + os.sep)
        ):
            current_project_other_changes.append(path)

            continue

        # ---------------------------------------------------------
        # 其他 project / 文件
        #
        # 例如：
        #
        # personDatabaseSegment/
        # wireDatabaseSegment/
        # cvat_dataset/
        # .idea/
        # README.md
        #
        # 不影响当前提交。
        # ---------------------------------------------------------

        unrelated_changes.append(path)

    # =========================================================================
    # 当前 project 其他修改
    # =========================================================================

    if current_project_other_changes:

        print("\n" + "=" * 80)
        print("警告：当前 project 存在其他未提交修改")
        print("=" * 80)

        for path in current_project_other_changes:
            print("  {}".format(path))

        print()
        print(
            "这些文件不会被本次提交自动加入。"
        )

    # =========================================================================
    # 其他 project 修改
    # =========================================================================

    if unrelated_changes:

        print("\n" + "=" * 80)
        print("其他项目 / 文件存在未提交修改")
        print("=" * 80)

        for path in unrelated_changes:
            print("  {}".format(path))

        print()
        print(
            "这些修改不会影响当前 project 的提交。"
        )

    print("\n工作区检查完成。")


# =============================================================================
# 文件检查
# =============================================================================

def check_dataset_exists(dataset_path):
    """
    检查数据批次目录是否存在。
    """

    if not os.path.isdir(dataset_path):
        raise RuntimeError(
            "数据批次目录不存在：\n{}".format(
                dataset_path
            )
        )

    print(
        "数据批次存在：{}".format(
            dataset_path
        )
    )


def check_dvc_file_exists(dvc_path):
    """
    检查 DVC 文件是否存在。
    """

    if not os.path.isfile(dvc_path):
        raise RuntimeError(
            "DVC 文件不存在：\n{}".format(
                dvc_path
            )
        )

    print(
        "DVC 文件存在：{}".format(
            dvc_path
        )
    )


# =============================================================================
# Git 暂存区检查
# =============================================================================

def get_staged_files():
    """
    获取 Git 暂存区中的文件。
    """

    result = subprocess.run(
        [
            "git",
            "diff",
            "--cached",
            "--name-only",
        ],
        cwd=REPO_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "获取 Git 暂存区文件失败：\n{}".format(
                result.stderr
            )
        )

    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def check_staged_files(expected_files):
    """
    检查暂存区中是否出现非预期文件。

    expected_files：
        本次任务允许提交的文件列表。

    如果发现其他文件已经被 staged，
    为避免误提交，程序直接停止。
    """

    staged_files = get_staged_files()

    expected_files_set = set(
        os.path.normpath(path)
        for path in expected_files
    )

    staged_files_set = set(
        os.path.normpath(path)
        for path in staged_files
    )

    unexpected_files = (
            staged_files_set - expected_files_set
    )

    if unexpected_files:

        print("\n" + "=" * 80)
        print("错误：Git 暂存区存在非本次任务预期文件")
        print("=" * 80)

        for path in sorted(unexpected_files):
            print("  {}".format(path))

        print("=" * 80)

        raise RuntimeError(
            "暂存区包含非预期文件，为避免误提交，程序停止。"
        )

    print("\nGit 暂存区检查通过。")

    if staged_files:
        print("本次准备提交的文件：")

        for path in staged_files:
            print("  {}".format(path))

    else:
        print("Git 暂存区为空。")


# =============================================================================
# Git Add
# =============================================================================

def git_add_files(files):
    """
    添加指定文件到 Git 暂存区。

    只添加传入的文件，不执行：

        git add .

    因此不会把其他项目文件加入 commit。
    """

    relative_files = []

    for path in files:

        if os.path.isabs(path):

            relative_path = os.path.relpath(
                path,
                REPO_DIR,
            )

        else:

            relative_path = path

        relative_files.append(
            relative_path
        )

    if not relative_files:
        raise RuntimeError(
            "没有需要 git add 的文件。"
        )

    run_command(
        [
            "git",
            "add",
            "--",
        ] + relative_files
    )

    return relative_files


# =============================================================================
# 提交数据批次
# =============================================================================

def commit_dataset_batch(project, batch):
    config = PROJECT_CONFIG[project]

    branch = config["branch"]

    dataset_dir = config["dataset_dir"]

    # -------------------------------------------------------------------------
    # 数据批次目录
    # -------------------------------------------------------------------------

    batch_dir = os.path.join(
        dataset_dir,
        batch,
    )

    dataset_path = os.path.join(
        REPO_DIR,
        batch_dir,
    )

    # -------------------------------------------------------------------------
    # DVC 文件
    # -------------------------------------------------------------------------

    dvc_file = os.path.join(
        REPO_DIR,
        dataset_dir,
        "{}.dvc".format(batch),
    )

    # -------------------------------------------------------------------------
    # 增量信息
    # -------------------------------------------------------------------------

    increment_info_file = os.path.join(
        REPO_DIR,
        dataset_dir,
        "dataset_increment_info.json",
    )

    # -------------------------------------------------------------------------
    # 打印任务信息
    # -------------------------------------------------------------------------

    print("\n")
    print("#" * 80)
    print("# Git + DVC 数据批次提交")
    print("#" * 80)

    print(
        "Project       : {}".format(
            project
        )
    )

    print(
        "Git Branch    : {}".format(
            branch
        )
    )

    print(
        "Dataset Dir   : {}".format(
            dataset_dir
        )
    )

    print(
        "Batch         : {}".format(
            batch
        )
    )

    print(
        "Batch Path    : {}".format(
            batch_dir
        )
    )

    print(
        "DVC File      : {}".format(
            dvc_file
        )
    )

    print(
        "Increment Info: {}".format(
            increment_info_file
        )
    )

    print("#" * 80)

    # =========================================================================
    # [1/6] 检查数据批次
    # =========================================================================

    print("\n[1/6] 检查数据批次目录...")

    check_dataset_exists(
        dataset_path
    )

    # =========================================================================
    # [2/6] 获取当前 branch
    # =========================================================================

    print("\n[2/6] 检查 Git Branch...")

    current_branch = get_current_branch()

    print(
        "当前 Git Branch：{}".format(
            current_branch
        )
    )

    print(
        "目标 Git Branch：{}".format(
            branch
        )
    )

    # =========================================================================
    # 如果需要切换 branch
    #
    # 这里仍然要求整个工作区 clean。
    #
    # 原因：
    # checkout 可能覆盖其他未提交修改。
    # =========================================================================

    if current_branch != branch:

        print("\n当前不在目标 branch，需要切换。")

        print(
            "为了避免切换 branch 覆盖其他未提交修改，"
            "本次切换要求 Git 工作区 clean。"
        )

        check_git_clean_for_branch_switch()

        run_command(
            [
                "git",
                "checkout",
                branch,
            ]
        )

        current_branch = get_current_branch()

        if current_branch != branch:
            raise RuntimeError(
                "Git branch 切换失败。\n"
                "当前 branch：{}\n"
                "目标 branch：{}".format(
                    current_branch,
                    branch,
                )
            )

    else:

        print(
            "已经位于目标 branch，无需切换。"
        )

    # =========================================================================
    # [3/6] 检查当前 project / batch 修改
    # =========================================================================

    print(
        "\n[3/6] 检查当前 project / batch Git 状态..."
    )

    check_current_project_changes(
        project,
        batch,
    )

    # =========================================================================
    # [4/6] DVC Add
    # =========================================================================

    print("\n[4/6] 执行 DVC Add...")

    run_command(
        [
            "dvc",
            "add",
            batch_dir,
        ]
    )

    # =========================================================================
    # 检查 DVC 文件
    # =========================================================================

    check_dvc_file_exists(
        dvc_file
    )

    # =========================================================================
    # Git Add
    # =========================================================================

    print("\n执行 Git Add...")

    git_add_files_list = [
        ".gitignore",
        dvc_file,
    ]

    # dataset_increment_info.json 存在时加入
    if os.path.isfile(
            increment_info_file
    ):
        git_add_files_list.append(
            increment_info_file
        )

    # 转换成相对路径
    git_add_files_relative = []

    for path in git_add_files_list:

        if os.path.isabs(path):

            relative_path = os.path.relpath(
                path,
                REPO_DIR,
            )

        else:

            relative_path = path

        git_add_files_relative.append(
            relative_path
        )

    print("\n本次 Git Add 文件：")

    for path in git_add_files_relative:
        print(
            "  {}".format(path)
        )

    # -------------------------------------------------------------------------
    # 只添加指定文件
    # -------------------------------------------------------------------------

    git_add_files(
        git_add_files_relative
    )

    # =========================================================================
    # [5/6] 检查 Git 状态 / 暂存区
    # =========================================================================

    print(
        "\n[5/6] 检查 Git 暂存区..."
    )

    run_command(
        [
            "git",
            "status",
            "--short",
        ]
    )

    # -------------------------------------------------------------------------
    # 非预期 staged 文件检查
    # -------------------------------------------------------------------------

    check_staged_files(
        git_add_files_relative
    )

    # =========================================================================
    # Commit Message
    # =========================================================================

    commit_message = (
        "project:{}\n\n"
        "Data batch:{}\n\n"
        "add {}"
    ).format(
        project,
        batch,
        batch_dir,
    )

    print("\nCommit Message：")
    print("-" * 80)
    print(commit_message)
    print("-" * 80)

    # =========================================================================
    # [6/6] Git Commit
    # =========================================================================

    print("\n[6/6] 执行 Git Commit...")

    run_command(
        [
            "git",
            "commit",
            "-m",
            commit_message,
        ]
    )

    # =========================================================================
    # 完成
    # =========================================================================

    print("\n")
    print("#" * 80)
    print("# 数据批次提交成功")
    print("#" * 80)

    print(
        "Project       : {}".format(
            project
        )
    )

    print(
        "Branch        : {}".format(
            branch
        )
    )

    print(
        "Batch         : {}".format(
            batch
        )
    )

    print(
        "DVC File      : {}".format(
            dvc_file
        )
    )

    print("#" * 80)


# =============================================================================
# 参数解析
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Git + DVC 数据批次自动提交脚本"
        )
    )

    parser.add_argument(
        "--project",
        required=True,
        choices=PROJECT_CONFIG.keys(),
        help="数据项目",
    )

    parser.add_argument(
        "--batch",
        required=True,
        help="数据批次，例如：date_20260928",
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    project = args.project
    batch = args.batch

    try:

        commit_dataset_batch(
            project,
            batch,
        )

    except Exception as e:

        print("\n")
        print("#" * 80)
        print("# 数据批次提交失败")
        print("#" * 80)

        print(
            str(e)
        )

        print("#" * 80)

        sys.exit(1)


if __name__ == "__main__":
    main()

"""


### 使用方式

你现在的 carpet 批次：

```bash
python 3.git_dataset_commit.py \
    --project carpet \
    --batch date_20260928
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
