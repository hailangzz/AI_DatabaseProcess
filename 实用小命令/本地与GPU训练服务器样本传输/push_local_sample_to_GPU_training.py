import shutil
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
backup_files = 0

lock = threading.Lock()

thread_local = threading.local()


# ==========================
# 文件大小格式化
# ==========================


def sizeof_fmt(num):
    for unit in ["B", "KB", "MB", "GB", "TB"]:

        if num < 1024:
            return f"{num:.2f} {unit}"

        num /= 1024

    return f"{num:.2f} PB"


# ==========================
# 获取所有文件
# ==========================


def get_all_upload_files(local_path):
    local_path = Path(local_path)

    upload_files = []

    for file_path in local_path.rglob("*"):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() in ALLOW_SUFFIXES:
            upload_files.append(file_path)

    return upload_files


# ==========================
# SSH连接池
# ==========================


def get_ssh_client(hostname, username, password, port):
    if not hasattr(thread_local, "ssh"):
        ssh = SSHClient()

        ssh.load_system_host_keys()

        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

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

        thread_local.ssh = ssh

    return thread_local.ssh


# ==========================
# 判断远程文件存在
# ==========================


def remote_file_exists(ssh, remote_file_path):
    cmd = f'test -f "{remote_file_path}" ' f"&& echo exists"

    stdin, stdout, stderr = ssh.exec_command(cmd)

    result = stdout.read().decode().strip()

    return result == "exists"


# ==========================
# 单文件上传
# ==========================


def upload_one_file(
        file_path,
        local_root,
        remote_root,
        hostname,
        username,
        password,
        port,
        backup_root=None,
):
    global uploaded_bytes
    global uploaded_files
    global skipped_files
    global backup_files

    try:

        ssh = get_ssh_client(hostname, username, password, port)

        # 相对路径

        relative_path = file_path.relative_to(local_root)

        # =====================
        # 远程路径
        # =====================

        remote_file = Path(remote_root) / relative_path

        remote_file = str(remote_file)

        remote_dir = str(Path(remote_file).parent)

        ssh.exec_command(f'mkdir -p "{remote_dir}"')

        # =====================
        # 判断远程是否存在
        # =====================

        if remote_file_exists(ssh, remote_file):
            with lock:
                skipped_files += 1

            return

        # =====================
        # 上传
        # =====================

        with SCPClient(ssh.get_transport()) as scp:

            scp.put(str(file_path), remote_path=remote_file)

        # =====================
        # 本地备份
        # =====================

        if backup_root is not None:
            backup_file = Path(backup_root) / relative_path

            backup_file.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(file_path, backup_file)

            with lock:
                backup_files += 1

        # =====================
        # 更新统计
        # =====================

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
                flush=True,
            )

    except Exception as e:

        print(f"\n上传失败: {file_path}\n{e}")


# ==========================
# 主上传函数
# ==========================


def upload_to_server(
        local_path,
        remote_path,
        hostname,
        username,
        password=None,
        port=22,
        max_workers=16,
        backup_path=None,
):
    global total_bytes

    local_path = Path(local_path)

    upload_files = get_all_upload_files(local_path)

    total_bytes = sum(f.stat().st_size for f in upload_files)

    print(f"待上传文件数量: {len(upload_files)}")

    print(f"总大小: {sizeof_fmt(total_bytes)}")

    print(f"线程数: {max_workers}")

    if backup_path:
        print(f"本地备份目录: {backup_path}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

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
                port,
                backup_path,
            )

            futures.append(future)

        for future in as_completed(futures):
            future.result()

    print("\n\n上传完成")

    print(f"实际上传文件: {uploaded_files}")

    print(f"跳过文件: {skipped_files}")

    print(f"本地备份文件: {backup_files}")


# ==========================
# main
# ==========================


if __name__ == "__main__":
    hostname = "172.16.50.229"
    username = "robot-server"
    password = "black@box"

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

    # # ==========================
    # # 塑料袋 images
    # # ==========================
    #
    # # ==========================
    # # 塑料袋 labels
    # # ==========================
    # local_images_path = "/data/database/Total_auto_augmentor_database/plasticbagDatabaseAugmentor/date0702_public/images"
    # local_labels_path = "/data/database/Total_auto_augmentor_database/plasticbagDatabaseAugmentor/date0702_public/labels"
    #
    # remote_images_path = ("/home/robot-server/data/AITotal_SegmentDatabase/plasticbagDatabaseSegment/images/train")
    # remote_labels_path = ("/home/robot-server/data/AITotal_SegmentDatabase/plasticbagDatabaseSegment/labels/train")
    #
    # backup_images_path = "/data/database/AITotal_SegmentDatabase/plasticbagDatabaseSegment/images/train"
    # backup_labels_path = "/data/database/AITotal_SegmentDatabase/plasticbagDatabaseSegment/labels/train"

    # ==========================
    # 行人 images
    # ==========================

    # ==========================
    # 行人 labels
    # ==========================
    local_images_path = "/data/database/Total_auto_augmentor_database/plasticbagDatabaseAugmentor/date0702_public/images"
    local_labels_path = "/data/database/Total_auto_augmentor_database/plasticbagDatabaseAugmentor/date0702_public/labels"

    remote_images_path = ("/home/robot-server/data/AITotal_SegmentDatabase/plasticbagDatabaseSegment/images/train")
    remote_labels_path = ("/home/robot-server/data/AITotal_SegmentDatabase/plasticbagDatabaseSegment/labels/train")

    backup_images_path = "/data/database/AITotal_SegmentDatabase/plasticbagDatabaseSegment/images/train"
    backup_labels_path = "/data/database/AITotal_SegmentDatabase/plasticbagDatabaseSegment/labels/train"

    upload_to_server(
        local_path=local_images_path,
        remote_path=remote_images_path,
        hostname=hostname,
        username=username,
        password=password,
        max_workers=8,
        backup_path=backup_images_path,
    )

    upload_to_server(
        local_path=local_labels_path,
        remote_path=remote_labels_path,
        hostname=hostname,
        username=username,
        password=password,
        max_workers=8,
        backup_path=backup_labels_path,
    )
