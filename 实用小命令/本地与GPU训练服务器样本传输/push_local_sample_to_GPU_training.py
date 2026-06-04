import os
from pathlib import Path

import paramiko
from paramiko import SSHClient
from scp import SCPClient


# 允许上传的后缀
ALLOW_SUFFIXES = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".txt"
]


# 全局进度变量
uploaded_bytes = 0
total_bytes = 0
current_index = 0
total_files = 0


def sizeof_fmt(num):
    """
    文件大小格式化
    """

    for unit in ["B", "KB", "MB", "GB", "TB"]:

        if num < 1024:
            return f"{num:.2f} {unit}"

        num /= 1024

    return f"{num:.2f} PB"


def progress(filename, size, sent):
    """
    当前文件上传进度
    """

    global uploaded_bytes

    if isinstance(filename, bytes):
        filename = filename.decode()

    current_total = uploaded_bytes + sent

    # percent = current_total / total_bytes * 100
    if total_bytes <= 0:
        percent = 0
    else:
        percent = current_total / total_bytes * 100

    print(
        f"\r[{current_index}/{total_files}] "
        f"{os.path.basename(filename)} "
        f"| {sizeof_fmt(current_total)} / {sizeof_fmt(total_bytes)} "
        f"| {percent:.2f}%",
        end="",
        flush=True
    )


def get_all_upload_files(local_path):
    """
    获取待上传文件
    """

    local_path = Path(local_path)

    upload_files = []

    for file_path in local_path.rglob("*"):

        if file_path.is_file():
            if "images" in str(local_path):    # 上传图像样本
                if file_path.suffix.lower() in ALLOW_SUFFIXES[:-1]:
                    upload_files.append(file_path)
                    # print(file_path)
            elif "labels" in  str(local_path): # 上传标注样本
                if file_path.suffix.lower() in ALLOW_SUFFIXES[-1:]:
                    upload_files.append(file_path)
                    # print(file_path)

    return upload_files


def remote_file_exists(ssh, remote_file_path):
    """
    检查远程文件是否存在
    """

    cmd = f'test -f "{remote_file_path}" && echo exists'

    stdin, stdout, stderr = ssh.exec_command(cmd)

    result = stdout.read().decode().strip()

    return result == "exists"


def upload_to_server(
    local_path,
    remote_path,
    hostname,
    username,
    password=None,
    port=22,
):

    global uploaded_bytes
    global total_bytes
    global current_index
    global total_files

    ssh = SSHClient()

    ssh.load_system_host_keys()

    ssh.set_missing_host_key_policy(
        paramiko.AutoAddPolicy()
    )

    print(f"正在连接服务器 {hostname} ...")

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

        transport = ssh.get_transport()

        if transport is None or not transport.is_active():
            raise Exception("SSH transport 未建立成功")

        print("服务器连接成功")

        # 获取上传文件
        upload_files = get_all_upload_files(local_path)

        total_files = len(upload_files)

        # 统计总大小
        total_bytes = sum(
            file.stat().st_size
            for file in upload_files
        )

        print(f"待上传文件数量: {total_files}")

        print(f"总大小: {sizeof_fmt(total_bytes)}")

        local_path = Path(local_path)

        skipped_files = 0
        uploaded_files = 0

        with SCPClient(
            transport,
            progress=progress
        ) as scp:

            for idx, file_path in enumerate(upload_files):

                current_index = idx + 1

                relative_path = (
                    file_path.relative_to(local_path)
                )

                remote_file_path = (
                    Path(remote_path) / relative_path
                )

                remote_dir = str(
                    remote_file_path.parent
                )

                # 创建远程目录
                ssh.exec_command(
                    f'mkdir -p "{remote_dir}"'
                )

                # =========================
                # 检查远程文件是否已存在
                # =========================
                if remote_file_exists(
                    ssh,
                    str(remote_file_path)
                ):

                    # print(
                    #     f"\n跳过已存在文件: "
                    #     f"{relative_path}"
                    # )

                    skipped_files += 1

                    continue

                # =========================
                # 上传文件
                # =========================
                scp.put(
                    str(file_path),
                    remote_path=str(remote_file_path)
                )

                uploaded_bytes += (
                    file_path.stat().st_size
                )

                uploaded_files += 1

        print("\n\n上传完成！")

        print(f"实际上传文件数量: {uploaded_files}")

        print(f"跳过文件数量: {skipped_files}")

    except Exception as e:

        print(f"\n上传失败: {e}")

    finally:

        ssh.close()


if __name__ == "__main__":

    # 线材检测样本它，推送
    # local_path = "/data/database/AITotal_Real_Customer_Database/Real_Wire_Customer_Database/date0602_2/images"
    # remote_path = "/home/robot-server/data/AITotal_SegmentDatabase/wireDatabaseSegment/images/train"

    # local_path = "/data/database/AITotal_Real_Customer_Database/Real_Wire_Customer_Database/date0602_2/yolov8_labels/seg"
    # remote_path = "/home/robot-server/data/AITotal_SegmentDatabase/wireDatabaseSegment/labels/train"


    # 全部线材样本
    # local_path = "/data/database/AITotal_Real_Customer_Database/Real_Wire_Customer_Database/date0602_1/segment_database_augmentor_0602_batch1/images"    #
    # remote_path = "/home/robot-server/data/AITotal_SegmentDatabase/wireDatabaseSegment_all_database/images/train"
    #
    local_path = "/data/database/AITotal_Real_Customer_Database/Real_Wire_Customer_Database/date0602_1/segment_database_augmentor_0602_batch1/labels"
    remote_path = "/home/robot-server/data/AITotal_SegmentDatabase/wireDatabaseSegment_all_database/labels/train"



    # 污渍检测样本它，推送

    # local_path = "/data/database/AITotal_Real_Customer_Database/Real_Liquid_Customer_Database/images"    #
    # remote_path = "/home/robot-server/data/AITotal_SegmentDatabase/liquidDatabaseSegment/images/train"

    # local_path = "/data/database/AITotal_Real_Customer_Database/Real_Liquid_Customer_Database/labels"
    # remote_path = "/home/robot-server/data/AITotal_SegmentDatabase/liquidDatabaseSegment/labels/train"

    hostname = "172.16.50.229"
    username = "robot-server"
    password = "black@box"

    upload_to_server(
        local_path=local_path,
        remote_path=remote_path,
        hostname=hostname,
        username=username,
        password=password,
    )