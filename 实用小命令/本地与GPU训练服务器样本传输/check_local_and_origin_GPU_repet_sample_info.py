import paramiko
from paramiko import SSHClient
from pathlib import Path


# 图像后缀
IMAGE_SUFFIXES = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
]


def get_local_files(
    local_path,
    file_type="image"
):
    """
    获取本地文件名集合
    """

    path = Path(local_path)

    file_set = set()

    if file_type == "image":

        suffixes = IMAGE_SUFFIXES

    elif file_type == "txt":

        suffixes = [".txt"]

    else:

        raise ValueError("file_type 必须为 image 或 txt")

    for suffix in suffixes:

        for file_path in path.rglob(f"*{suffix}"):

            file_set.add(file_path.stem)

    return file_set


def get_remote_files(
    ssh,
    remote_path,
    file_type="image"
):
    """
    获取远程文件名集合
    """

    if file_type == "image":

        cmd = (
            f'find "{remote_path}" -type f | '
            f'grep -Ei "\\.(jpg|jpeg|png|bmp|webp)$"'
        )

    elif file_type == "txt":

        cmd = (
            f'find "{remote_path}" -type f | '
            f'grep -Ei "\\.txt$"'
        )

    else:

        raise ValueError("file_type 必须为 image 或 txt")

    stdin, stdout, stderr = ssh.exec_command(cmd)

    files = stdout.read().decode().splitlines()

    file_set = set()

    for file in files:

        stem = Path(file).stem

        file_set.add(stem)

    return file_set


def compare_local_and_remote(
    local_path,
    remote_path,
    hostname,
    username,
    password,
    file_type="image",
    port=22,
):

    print(f"\n===== 开始检查 {file_type} 文件 =====\n")

    # 获取本地文件
    local_files = get_local_files(
        local_path,
        file_type
    )

    print(f"本地文件数量: {len(local_files)}")

    # SSH连接
    ssh = SSHClient()

    ssh.set_missing_host_key_policy(
        paramiko.AutoAddPolicy()
    )

    print(f"\n正在连接服务器 {hostname} ...")

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

    print("服务器连接成功")

    # 获取远程文件
    remote_files = get_remote_files(
        ssh,
        remote_path,
        file_type
    )

    print(f"远程文件数量: {len(remote_files)}")

    # 同名文件
    same_files = local_files & remote_files

    # 本地独有
    only_local = local_files - remote_files

    # 远程独有
    only_remote = remote_files - local_files

    print("\n===== 对比结果 =====\n")

    print(f"同名文件数量: {len(same_files)}")

    print(f"仅本地存在文件数量: {len(only_local)}")

    print(f"仅远程存在文件数量: {len(only_remote)}")

    ssh.close()

    return {
        "local_count": len(local_files),
        "remote_count": len(remote_files),
        "same_count": len(same_files),
        "only_local_count": len(only_local),
        "only_remote_count": len(only_remote),
    }


if __name__ == "__main__":

    # 本地路径
    local_path = "/data/database/AITotal_SegmentDatabase/wireDatabaseSegment/images/train"

    # 远程路径
    remote_path = "/home/robot-server/data/AITotal_SegmentDatabase/wireDatabaseSegment_all_database/images/train"
    # remote_path = "/home/robot-server/data/AITotal_SegmentDatabase/wireDatabaseSegment/images/train"

    hostname = "172.16.50.229"
    username = "robot-server"
    password = "black@box"

    compare_local_and_remote(
        local_path=local_path,
        remote_path=remote_path,
        hostname=hostname,
        username=username,
        password=password,
        file_type="image",  # image 或 txt
    )