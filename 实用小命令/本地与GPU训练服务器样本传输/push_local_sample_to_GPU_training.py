import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import paramiko
from paramiko import SSHClient
from scp import SCPClient

ALLOW_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".txt"]

uploaded_bytes = 0
total_bytes = 0
uploaded_files = 0
skipped_files = 0

lock = threading.Lock()

thread_local = threading.local()


def sizeof_fmt(num):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024:
            return f"{num:.2f} {unit}"
        num /= 1024
    return f"{num:.2f} PB"


def get_all_upload_files(local_path):
    local_path = Path(local_path)

    upload_files = []

    for file_path in local_path.rglob("*"):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() in ALLOW_SUFFIXES:
            upload_files.append(file_path)

    return upload_files


def get_ssh_client(
        hostname,
        username,
        password,
        port):
    if not hasattr(thread_local, "ssh"):
        ssh = SSHClient()

        ssh.load_system_host_keys()

        ssh.set_missing_host_key_policy(
            paramiko.AutoAddPolicy()
        )

        ssh.connect(
            hostname=hostname,
            port=port,
            username=username,
            password=password,
            timeout=10,
            banner_timeout=30,
            auth_timeout=30,
            look_for_keys=False,
            allow_agent=False
        )

        thread_local.ssh = ssh

    return thread_local.ssh


def remote_file_exists(ssh, remote_file_path):
    cmd = f'test -f "{remote_file_path}" && echo exists'

    stdin, stdout, stderr = ssh.exec_command(cmd)

    result = stdout.read().decode().strip()

    return result == "exists"


def upload_one_file(
        file_path,
        local_root,
        remote_root,
        hostname,
        username,
        password,
        port):
    global uploaded_bytes
    global uploaded_files
    global skipped_files

    try:

        ssh = get_ssh_client(
            hostname,
            username,
            password,
            port
        )

        relative_path = file_path.relative_to(local_root)

        remote_file = Path(remote_root) / relative_path

        remote_file = str(remote_file)

        remote_dir = str(Path(remote_file).parent)

        ssh.exec_command(
            f'mkdir -p "{remote_dir}"'
        )

        if remote_file_exists(ssh, remote_file):
            with lock:
                skipped_files += 1

            return

        with SCPClient(
                ssh.get_transport()) as scp:

            scp.put(
                str(file_path),
                remote_path=remote_file
            )

        file_size = file_path.stat().st_size

        with lock:

            uploaded_files += 1

            uploaded_bytes += file_size

            percent = uploaded_bytes / total_bytes * 100

            print(
                f"\r上传: "
                f"{uploaded_files} "
                f"| "
                f"{sizeof_fmt(uploaded_bytes)}"
                f"/"
                f"{sizeof_fmt(total_bytes)} "
                f"| {percent:.2f}%",
                end="",
                flush=True
            )

    except Exception as e:

        print(
            f"\n上传失败: {file_path}\n{e}"
        )


def upload_to_server(
        local_path,
        remote_path,
        hostname,
        username,
        password=None,
        port=22,
        max_workers=16):
    global total_bytes

    local_path = Path(local_path)

    upload_files = get_all_upload_files(local_path)

    total_bytes = sum(
        f.stat().st_size
        for f in upload_files
    )

    print(
        f"待上传文件数量: "
        f"{len(upload_files)}"
    )

    print(
        f"总大小: "
        f"{sizeof_fmt(total_bytes)}"
    )

    print(
        f"线程数: {max_workers}"
    )

    with ThreadPoolExecutor(
            max_workers=max_workers) as executor:

        futures = []

        for file_path in upload_files:
            future = executor.submit(
                upload_one_file,
                file_path,
                local_path,
                remote_path,
                hostname,
                username,
                password,
                port
            )

            futures.append(future)

        for future in as_completed(futures):
            future.result()

    print("\n\n上传完成")

    print(
        f"实际上传文件: "
        f"{uploaded_files}"
    )

    print(
        f"跳过文件: "
        f"{skipped_files}"
    )


if __name__ == "__main__":
    # real线材检测样本它，推送
    # local_path = "/data/database/AITotal_Real_Customer_Database/Real_Wire_Customer_Database/date0602_2/segment_database_augmentor_0602_batch2/images"
    # remote_path = "/home/robot-server/data/AITotal_SegmentDatabase/wireDatabaseSegment/images/train"

    # local_path = "/data/database/AITotal_Real_Customer_Database/Real_Wire_Customer_Database/date0602_2/segment_database_augmentor_0602_batch2/labels"
    # remote_path = "/home/robot-server/data/AITotal_SegmentDatabase/wireDatabaseSegment/labels/train"

    # 地毯检测

    # local_path = "/data/database/AITotal_Real_Customer_Database/Real_Carpet_Customer_Database/date0612/images"
    # remote_path = "/home/robot-server/data/AITotal_SegmentDatabase/carpetDatabaseSegment/images/train"

    # local_path = "/data/database/AITotal_Real_Customer_Database/Real_Carpet_Customer_Database/date0612/labels"
    # remote_path = "/home/robot-server/data/AITotal_SegmentDatabase/carpetDatabaseSegment/labels/train"

    # 全部线材样本
    # local_path = "/data/database/AITotal_Real_Customer_Database/Real_Wire_Customer_Database/date0602_1/segment_database_augmentor_0602_batch1/images"    #
    # remote_path = "/home/robot-server/data/AITotal_SegmentDatabase/wireDatabaseSegment_all_database/images/train"
    #
    # local_path = "/data/database/AITotal_Real_Customer_Database/Real_Wire_Customer_Database/date0602_1/segment_database_augmentor_0602_batch1/labels"
    # remote_path = "/home/robot-server/data/AITotal_SegmentDatabase/wireDatabaseSegment_all_database/labels/train"

    # 污渍检测样本，推送

    # local_path = "/data/database/Total_auto_augmentor_database/liquidDatabaseAugmentor/date0629/real_liquid/images"
    # remote_path = "/home/robot-server/data/AITotal_SegmentDatabase/liquidDatabaseSegment/images/train"

    # local_path = "/data/database/Total_auto_augmentor_database/liquidDatabaseAugmentor/date0629/real_liquid/labels"
    # remote_path = "/home/robot-server/data/AITotal_SegmentDatabase/liquidDatabaseSegment/labels/train"

    # 塑料袋检测样本，推送
    # local_path = "/data/database/Total_auto_augmentor_database/plasticbagDatabaseAugmentor/date0702_public/images"
    # remote_path = "/home/robot-server/data/AITotal_SegmentDatabase/plasticbagDatabaseSegment/images/train"

    local_path = "/data/database/Total_auto_augmentor_database/plasticbagDatabaseAugmentor/date0702_public/labels"
    remote_path = "/home/robot-server/data/AITotal_SegmentDatabase/plasticbagDatabaseSegment/labels/train"

    hostname = "172.16.50.229"
    username = "robot-server"
    password = "black@box"

    upload_to_server(
        local_path=local_path,
        remote_path=remote_path,
        hostname=hostname,
        username=username,
        password=password,
        max_workers=8
    )
