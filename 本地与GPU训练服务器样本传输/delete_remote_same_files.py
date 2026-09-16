# -*- coding: utf-8 -*-

"""
本地 / 远程文件同名检查 + 删除远程同名文件

功能：
1. 扫描本地指定目录
2. 扫描远程指定目录
3. 支持 image / txt
4. 默认非递归，只扫描当前目录
5. 使用 --recursive 后递归扫描所有子目录
6. 找出本地和远程同名文件
7. 删除远程同名文件
8. 支持 --dry-run 预览删除结果
9. 支持 --yes 跳过删除确认

------------------------------------------------------------
使用示例
------------------------------------------------------------

1. 非递归检查 txt：

python delete_remote_same_files.py \
    --local-path /data/database/coco2017_non_person/labels \
    --remote-path /home/robot-server/data/AITotal_SegmentDatabase/personDatabaseSegment/labels/train \
    --file-type txt

------------------------------------------------------------

2. 递归检查 txt：

python delete_remote_same_files.py \
    --local-path /data/database/coco2017_non_person/labels \
    --remote-path /home/robot-server/data/AITotal_SegmentDatabase/personDatabaseSegment/labels/train \
    --file-type txt \
    --recursive

------------------------------------------------------------

3. 递归检查 image：

python delete_remote_same_files.py \
    --local-path /data/database/coco2017_non_person/images \
    --remote-path /home/robot-server/data/AITotal_SegmentDatabase/personDatabaseSegment/images/train \
    --file-type image \
    --recursive

------------------------------------------------------------

4. 强烈推荐第一次使用 dry-run：

python delete_remote_same_files.py \
    --local-path /data/database/coco2017_non_person/labels \
    --remote-path /home/robot-server/data/AITotal_SegmentDatabase/personDatabaseSegment/labels/train \
    --file-type txt \
    --recursive \
    --dry-run

此时只显示将删除哪些文件，不会真正删除。

------------------------------------------------------------

5. 正式删除，并跳过确认：

python delete_remote_same_files.py \
    --local-path /data/database/coco2017_non_person/labels \
    --remote-path /home/robot-server/data/AITotal_SegmentDatabase/personDatabaseSegment/labels/train \
    --file-type txt \
    --recursive \
    --yes
"""

import argparse
import shlex
from pathlib import Path

import paramiko
from paramiko import SSHClient

# ============================================================
# 图像后缀
# ============================================================

IMAGE_SUFFIXES = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
]


# ============================================================
# 获取本地文件
# ============================================================


def get_local_files(
        local_path,
        file_type="image",
        recursive=False,
):
    """
    获取本地文件。

    参数：
        local_path:
            本地目录

        file_type:
            image / txt

        recursive:
            False:
                只扫描当前目录

            True:
                递归扫描所有子目录

    返回：
        dict

        {
            "abc": "/data/images/abc.jpg",
            "def": "/data/images/def.jpg",
        }

    key：
        文件 stem

    value：
        文件完整路径

    注意：
        如果存在同 stem 的多个文件，
        后扫描到的文件会覆盖前面的文件。
    """

    path = Path(local_path)

    if not path.exists():
        raise FileNotFoundError(f"本地路径不存在: {local_path}")

    if not path.is_dir():
        raise NotADirectoryError(f"本地路径不是目录: {local_path}")

    # --------------------------------------------------------
    # 文件类型
    # --------------------------------------------------------

    if file_type == "image":

        suffixes = IMAGE_SUFFIXES

    elif file_type == "txt":

        suffixes = [".txt"]

    else:

        raise ValueError("file_type 必须为 image 或 txt")

    # --------------------------------------------------------
    # 扫描文件
    # --------------------------------------------------------

    file_dict = {}

    if recursive:

        # 递归
        iterator = path.rglob("*")

    else:

        # 非递归
        iterator = path.glob("*")

    for file_path in iterator:

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in suffixes:
            continue

        stem = file_path.stem

        file_dict[stem] = str(file_path)

    return file_dict


# ============================================================
# 获取远程文件
# ============================================================


def get_remote_files(
        ssh,
        remote_path,
        file_type="image",
        recursive=False,
):
    """
    获取远程文件。

    返回：

        {
            "abc": "/remote/path/abc.jpg",
            "def": "/remote/path/def.jpg"
        }

    recursive=False：

        只扫描 remote_path 当前目录。

    recursive=True：

        扫描 remote_path 及其所有子目录。
    """

    # --------------------------------------------------------
    # 检查文件类型
    # --------------------------------------------------------

    if file_type == "image":

        suffix_pattern = r"\.(jpg|jpeg|png|bmp|webp)$"

    elif file_type == "txt":

        suffix_pattern = r"\.txt$"

    else:

        raise ValueError("file_type 必须为 image 或 txt")

    # --------------------------------------------------------
    # 构造 find 命令
    # --------------------------------------------------------

    if recursive:

        # 递归
        cmd = f'find "{remote_path}" ' f"-type f | " f'grep -Ei "{suffix_pattern}"'

    else:

        # 非递归
        #
        # -maxdepth 1：
        # 只扫描当前目录
        #
        # -mindepth 1：
        # 排除 remote_path 本身
        cmd = (
            f'find "{remote_path}" '
            f"-maxdepth 1 "
            f"-mindepth 1 "
            f"-type f | "
            f'grep -Ei "{suffix_pattern}"'
        )

    print()
    print(f"远程扫描模式: " f"{'递归' if recursive else '非递归'}")

    # --------------------------------------------------------
    # 执行 SSH 命令
    # --------------------------------------------------------

    stdin, stdout, stderr = ssh.exec_command(cmd)

    files = stdout.read().decode("utf-8", errors="ignore").splitlines()

    error = stderr.read().decode("utf-8", errors="ignore")

    if error.strip():
        print()
        print("[WARNING] 远程扫描出现错误:")

        print(error)

    # --------------------------------------------------------
    # 构造文件字典
    # --------------------------------------------------------

    file_dict = {}

    for file in files:

        file = file.strip()

        if not file:
            continue

        stem = Path(file).stem

        file_dict[stem] = file

    return file_dict


# ============================================================
# 删除远程文件
# ============================================================


def delete_remote_files(
        ssh,
        remote_files,
        same_files,
        dry_run=False,
        batch_size=1000,
):
    """
    批量删除远程同名文件。

    相比逐文件执行：

        ssh.exec_command("rm -f xxx")
        ssh.exec_command("rm -f xxx")
        ssh.exec_command("rm -f xxx")

    本函数采用：

        rm -f xxx xxx xxx ...

    并且按照 batch_size 分批执行。

    参数：

        ssh:
            Paramiko SSHClient

        remote_files:
            {
                "001": "/remote/path/001.txt",
                "002": "/remote/path/002.txt",
                ...
            }

        same_files:
            本地和远程同名的 stem 集合

        dry_run:
            True:
                只显示，不删除

        batch_size:
            每批删除多少文件
    """

    # ========================================================
    # 获取需要删除的远程文件
    # ========================================================

    delete_files = []

    for name in sorted(same_files):

        remote_file = remote_files.get(name)

        if remote_file:
            delete_files.append(
                remote_file
            )

    # ========================================================
    # 没有文件
    # ========================================================

    if not delete_files:
        print()
        print(
            "没有需要删除的远程文件。"
        )

        return {
            "deleted_count": 0,
            "failed_count": 0,
        }

    # ========================================================
    # 显示信息
    # ========================================================

    print()
    print("=" * 80)

    if dry_run:

        print(
            "DRY-RUN 模式"
        )

    else:

        print(
            "批量删除模式"
        )

    print("=" * 80)

    print(
        f"待删除文件数量: "
        f"{len(delete_files)}"
    )

    print(
        f"Batch Size      : "
        f"{batch_size}"
    )

    batch_count = (
                          len(delete_files)
                          + batch_size
                          - 1
                  ) // batch_size

    print(
        f"批次数量        : "
        f"{batch_count}"
    )

    # ========================================================
    # Dry Run
    # ========================================================

    if dry_run:

        print()

        for index, remote_file in enumerate(
                delete_files,
                start=1
        ):
            print(
                f"[{index}/{len(delete_files)}] "
                f"{remote_file}"
            )

        print()
        print(
            "[DRY-RUN] "
            "不会删除任何文件。"
        )

        return {
            "deleted_count": 0,
            "failed_count": 0,
        }

    # ========================================================
    # 开始批量删除
    # ========================================================

    deleted_count = 0

    failed_count = 0

    print()
    print(
        "开始批量删除远程文件..."
    )

    # ========================================================
    # 分批
    # ========================================================

    for batch_index in range(
            0,
            len(delete_files),
            batch_size
    ):

        batch = delete_files[
                batch_index:
                batch_index + batch_size
                ]

        current_batch = (
                batch_index // batch_size
                + 1
        )

        print(
            f"\n"
            f"[Batch "
            f"{current_batch}/{batch_count}] "
            f"删除 {len(batch)} 个文件..."
        )

        # ----------------------------------------------------
        # shell quote
        # ----------------------------------------------------

        quoted_files = [
            shlex.quote(path)
            for path in batch
        ]

        # ----------------------------------------------------
        # 一次 rm 删除整个 batch
        # ----------------------------------------------------

        command = (
                "rm -f -- "
                + " ".join(quoted_files)
        )

        stdin, stdout, stderr = (
            ssh.exec_command(command)
        )

        # 等待命令完成
        exit_code = (
            stdout.channel.recv_exit_status()
        )

        error = (
            stderr.read()
            .decode(
                "utf-8",
                errors="ignore"
            )
            .strip()
        )

        # ----------------------------------------------------
        # 判断结果
        # ----------------------------------------------------

        if exit_code == 0:

            deleted_count += len(batch)

            print(
                f"[SUCCESS] "
                f"本批删除 {len(batch)} 个文件"
            )

        else:

            failed_count += len(batch)

            print(
                f"[ERROR] "
                f"本批删除失败"
            )

            if error:
                print(
                    f"错误信息: {error}"
                )

    # ========================================================
    # 最终结果
    # ========================================================

    print()
    print("=" * 80)

    print(
        "批量删除完成"
    )

    print(
        f"成功删除: "
        f"{deleted_count}"
    )

    print(
        f"删除失败: "
        f"{failed_count}"
    )

    print("=" * 80)

    return {
        "deleted_count": deleted_count,
        "failed_count": failed_count,
    }


# ============================================================
# 比较并删除
# ============================================================


def compare_and_delete_remote(
        local_path,
        remote_path,
        hostname,
        username,
        password,
        file_type="image",
        port=22,
        recursive=False,
        dry_run=False,
        auto_confirm=False,
):
    """
    本地 / 远程文件比较，并删除远程同名文件。
    """

    print()
    print("=" * 80)

    print(f"开始处理 {file_type} 文件")

    print(f"扫描模式: " f"{'递归' if recursive else '非递归'}")

    print(f"删除模式: " f"{'DRY-RUN' if dry_run else '正式删除'}")

    print("=" * 80)

    # ========================================================
    # 1. 扫描本地
    # ========================================================

    print()
    print("[1/5] 扫描本地目录")

    print(f"      {local_path}")

    local_files = get_local_files(
        local_path=local_path,
        file_type=file_type,
        recursive=recursive,
    )

    print()
    print(f"本地文件数量: " f"{len(local_files)}")

    # ========================================================
    # 2. SSH连接
    # ========================================================

    print()
    print("[2/5] 连接远程服务器")

    print(f"      {hostname}")

    ssh = SSHClient()

    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:

        ssh.connect(
            hostname=hostname,
            port=port,
            username=username,
            password=password,
            timeout=10,
            banner_timeout=30,
            auth_timeout=30,
            look_for_keys=False,
            allow_agent=False,
        )

        print("      服务器连接成功")

        # ====================================================
        # 3. 扫描远程
        # ====================================================

        print()
        print("[3/5] 扫描远程目录")

        print(f"      {remote_path}")

        remote_files = get_remote_files(
            ssh=ssh,
            remote_path=remote_path,
            file_type=file_type,
            recursive=recursive,
        )

        print()
        print(f"远程文件数量: " f"{len(remote_files)}")

        # ====================================================
        # 4. 对比
        # ====================================================

        print()
        print("[4/5] 对比文件")

        local_names = set(local_files.keys())

        remote_names = set(remote_files.keys())

        # ----------------------------------------------------
        # 同名
        # ----------------------------------------------------

        same_files = local_names & remote_names

        # ----------------------------------------------------
        # 仅本地
        # ----------------------------------------------------

        only_local = local_names - remote_names

        # ----------------------------------------------------
        # 仅远程
        # ----------------------------------------------------

        only_remote = remote_names - local_names

        print()
        print("=" * 80)

        print("对比结果")

        print("=" * 80)

        print(f"本地文件数量       : " f"{len(local_files)}")

        print(f"远程文件数量       : " f"{len(remote_files)}")

        print(f"同名文件数量       : " f"{len(same_files)}")

        print(f"仅本地存在数量     : " f"{len(only_local)}")

        print(f"仅远程存在数量     : " f"{len(only_remote)}")

        # ====================================================
        # 打印同名文件
        # ====================================================

        if same_files:

            print()
            print("-" * 80)

            print("同名文件")

            print("-" * 80)

            for name in sorted(same_files):
                local_file = local_files[name]

                remote_file = remote_files[name]

                print()
                print(f"名称: {name}")

                print(f"  本地  : {local_file}")

                print(f"  远程  : {remote_file}")

        else:

            print()
            print("没有发现同名文件。")

            return {
                "local_count": len(local_files),
                "remote_count": len(remote_files),
                "same_count": 0,
                "only_local_count": len(only_local),
                "only_remote_count": len(only_remote),
                "deleted_count": 0,
                "failed_count": 0,
            }

        # ====================================================
        # 5. 删除确认
        # ====================================================

        print()
        print("[5/5] 准备处理远程同名文件")

        # ----------------------------------------------------
        # dry-run 不需要确认
        # ----------------------------------------------------

        if dry_run:

            delete_result = delete_remote_files(
                ssh=ssh,
                remote_files=remote_files,
                same_files=same_files,
                dry_run=True,
            )

        else:

            # ------------------------------------------------
            # 自动确认
            # ------------------------------------------------

            if auto_confirm:

                print()
                print("--yes 已指定，" "跳过删除确认。")

                delete_result = delete_remote_files(
                    ssh=ssh,
                    remote_files=remote_files,
                    same_files=same_files,
                    dry_run=False,
                )

            # ------------------------------------------------
            # 手动确认
            # ------------------------------------------------

            else:

                print()
                print("WARNING:")

                print(f"即将删除远程目录中 " f"{len(same_files)} 个同名文件。")

                print("该操作不可恢复。")

                answer = input("\n确认删除吗？" "请输入 YES 继续: ")

                if answer.strip() == "YES" or answer.strip() == "yes":

                    delete_result = delete_remote_files(
                        ssh=ssh,
                        remote_files=remote_files,
                        same_files=same_files,
                        dry_run=False,
                    )

                else:

                    print()
                    print("用户取消删除。")

                    delete_result = {
                        "deleted_count": 0,
                        "failed_count": 0,
                    }

        # ====================================================
        # 返回结果
        # ====================================================

        return {
            "local_count": len(local_files),
            "remote_count": len(remote_files),
            "same_count": len(same_files),
            "only_local_count": len(only_local),
            "only_remote_count": len(only_remote),
            "deleted_count": delete_result["deleted_count"],
            "failed_count": delete_result["failed_count"],
        }

    finally:

        ssh.close()

        print()
        print("SSH连接已关闭")


# ============================================================
# argparse
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(description=("比较本地和远程文件，" "删除远程同名文件"))

    # --------------------------------------------------------
    # 本地路径
    # --------------------------------------------------------

    parser.add_argument("--local-path", required=True, help="本地目录")

    # --------------------------------------------------------
    # 远程路径
    # --------------------------------------------------------

    parser.add_argument("--remote-path", required=True, help="远程目录")

    # --------------------------------------------------------
    # 文件类型
    # --------------------------------------------------------

    parser.add_argument(
        "--file-type",
        choices=[
            "image",
            "txt",
        ],
        default="image",
        help=("文件类型：" "image 或 txt"),
    )

    # --------------------------------------------------------
    # SSH
    # --------------------------------------------------------

    parser.add_argument("--hostname", default="172.16.50.229", help="SSH服务器地址")

    parser.add_argument("--username", default="robot-server", help="SSH用户名")

    parser.add_argument("--password", default="black@box", help="SSH密码")

    parser.add_argument("--port", type=int, default=22, help="SSH端口")

    # --------------------------------------------------------
    # recursive
    # --------------------------------------------------------

    parser.add_argument(
        "--recursive", action="store_true", help=("递归扫描子目录。" "不指定时只扫描当前目录。")
    )

    # --------------------------------------------------------
    # dry-run
    # --------------------------------------------------------

    parser.add_argument(
        "--dry-run", action="store_true", help=("只显示将删除的远程文件，" "不执行删除。")
    )

    # --------------------------------------------------------
    # yes
    # --------------------------------------------------------

    parser.add_argument("--yes", action="store_true", help=("正式删除时跳过确认。"))

    return parser.parse_args()


# ============================================================
# main
# ============================================================


def main():
    args = parse_args()

    print()
    print("=" * 80)

    print("本地 / 远程同名文件清理工具")

    print("=" * 80)

    print(f"本地目录   : {args.local_path}")

    print(f"远程目录   : {args.remote_path}")

    print(f"文件类型   : {args.file_type}")

    print(f"扫描模式   : " f"{'递归' if args.recursive else '非递归'}")

    print(f"删除模式   : " f"{'DRY-RUN' if args.dry_run else '正式删除'}")

    print(f"服务器     : " f"{args.hostname}:{args.port}")

    # ========================================================
    # 执行
    # ========================================================

    result = compare_and_delete_remote(
        local_path=args.local_path,
        remote_path=args.remote_path,
        hostname=args.hostname,
        username=args.username,
        password=args.password,
        file_type=args.file_type,
        port=args.port,
        recursive=args.recursive,
        dry_run=args.dry_run,
        auto_confirm=args.yes,
    )

    # ========================================================
    # 最终统计
    # ========================================================

    print()
    print()
    print("=" * 80)

    print("最终统计")

    print("=" * 80)

    print(f"本地文件数量       : " f"{result['local_count']}")

    print(f"远程文件数量       : " f"{result['remote_count']}")

    print(f"同名文件数量       : " f"{result['same_count']}")

    print(f"仅本地存在数量     : " f"{result['only_local_count']}")

    print(f"仅远程存在数量     : " f"{result['only_remote_count']}")

    print(f"成功删除数量       : " f"{result['deleted_count']}")

    print(f"删除失败数量       : " f"{result['failed_count']}")

    print("=" * 80)


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    main()

"""
4. 强烈推荐第一次使用 dry-run：

python delete_remote_same_files.py \
    --local-path /data/database/coco2017_non_person/labels \
    --remote-path /home/robot-server/data/AITotal_SegmentDatabase/personDatabaseSegment/labels/train \
    --file-type txt \
    --dry-run
    
python delete_remote_same_files.py \
    --local-path /data/database/coco2017_non_person/images \
    --remote-path /home/robot-server/data/AITotal_SegmentDatabase/personDatabaseSegment/images/train \
    --file-type image \
    --dry-run

"""
