import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import paramiko
from paramiko import SSHClient
from scp import SCPClient

# ============================================================
# Configuration
# ============================================================

ALLOW_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".txt",
}

# ============================================================
# Thread Local
# ============================================================

thread_local = threading.local()

# ============================================================
# Global Statistics
# ============================================================

uploaded_bytes = 0
total_bytes = 0

uploaded_files = 0
skipped_files = 0
failed_files = 0

lock = threading.Lock()


# ============================================================
# Utility
# ============================================================

def sizeof_fmt(num):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024:
            return f"{num:.2f} {unit}"

        num /= 1024

    return f"{num:.2f} PB"


# ============================================================
# File Scan
# ============================================================

def get_all_upload_files(local_path):
    """
    Recursively find all files that need to be uploaded.
    """

    local_path = Path(local_path)

    return [
        file_path
        for file_path in local_path.rglob("*")
        if file_path.is_file()
           and file_path.suffix.lower() in ALLOW_SUFFIXES
    ]


# ============================================================
# SSH / SCP Connection
# ============================================================

def get_connection(
        hostname,
        username,
        password,
        port
):
    """
    Each worker thread maintains its own SSH + SCP connection.
    """

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
            allow_agent=False,
        )

        scp = SCPClient(
            ssh.get_transport()
        )

        thread_local.ssh = ssh
        thread_local.scp = scp

        # Cache directories created by this thread
        thread_local.created_dirs = set()

    return (
        thread_local.ssh,
        thread_local.scp,
        thread_local.created_dirs,
    )


# ============================================================
# Close Thread Connection
# ============================================================

def close_thread_connection():
    """
    Close SSH/SCP connection belonging to current worker thread.
    """

    scp = getattr(
        thread_local,
        "scp",
        None
    )

    ssh = getattr(
        thread_local,
        "ssh",
        None
    )

    try:

        if scp is not None:
            scp.close()

    except Exception:
        pass

    try:

        if ssh is not None:
            ssh.close()

    except Exception:
        pass

    if hasattr(thread_local, "scp"):
        del thread_local.scp

    if hasattr(thread_local, "ssh"):
        del thread_local.ssh

    if hasattr(thread_local, "created_dirs"):
        del thread_local.created_dirs


# ============================================================
# Remote File Check
# ============================================================

def remote_file_exists(
        ssh,
        remote_file_path
):
    """
    Check whether remote file already exists.
    """

    cmd = (
        f'test -f "{remote_file_path}" '
        f'&& echo exists'
    )

    stdin, stdout, stderr = ssh.exec_command(cmd)

    result = stdout.read().decode().strip()

    return result == "exists"


# ============================================================
# Ensure Remote Directory
# ============================================================

def ensure_remote_dir(
        ssh,
        remote_dir,
        created_dirs
):
    """
    Create remote directory only once per worker thread.
    """

    if remote_dir in created_dirs:
        return

    ssh.exec_command(
        f'mkdir -p "{remote_dir}"'
    )

    created_dirs.add(remote_dir)


# ============================================================
# Progress
# ============================================================

def print_progress():
    percent = 0.0

    if total_bytes > 0:
        percent = (
                uploaded_bytes
                / total_bytes
                * 100
        )

    print(
        f"\r上传: "
        f"{uploaded_files} "
        f"| "
        f"{sizeof_fmt(uploaded_bytes)}"
        f"/"
        f"{sizeof_fmt(total_bytes)} "
        f"| "
        f"{percent:.2f}%",
        end="",
        flush=True,
    )


# ============================================================
# Upload One File
# ============================================================

def upload_one_file(
        file_path,
        local_root,
        remote_root,
        hostname,
        username,
        password,
        port
):
    global uploaded_bytes
    global uploaded_files
    global skipped_files
    global failed_files

    try:

        # ----------------------------------------------------
        # Get thread-local connection
        # ----------------------------------------------------

        ssh, scp, created_dirs = get_connection(
            hostname=hostname,
            username=username,
            password=password,
            port=port,
        )

        # ----------------------------------------------------
        # Calculate relative path
        # ----------------------------------------------------

        relative_path = file_path.relative_to(
            local_root
        )

        remote_file = (
                Path(remote_root)
                / relative_path
        )

        remote_file = str(remote_file)

        remote_dir = str(
            Path(remote_file).parent
        )

        # ----------------------------------------------------
        # Ensure remote directory
        # ----------------------------------------------------

        ensure_remote_dir(
            ssh=ssh,
            remote_dir=remote_dir,
            created_dirs=created_dirs,
        )

        # ----------------------------------------------------
        # Skip existing file
        # ----------------------------------------------------

        if remote_file_exists(
                ssh,
                remote_file
        ):
            with lock:
                skipped_files += 1

            return

        # ----------------------------------------------------
        # Upload
        # ----------------------------------------------------

        scp.put(
            str(file_path),
            remote_path=remote_file,
        )

        # ----------------------------------------------------
        # Update statistics
        # ----------------------------------------------------

        file_size = file_path.stat().st_size

        with lock:

            uploaded_files += 1

            uploaded_bytes += file_size

            print_progress()

    except Exception as e:

        with lock:
            failed_files += 1

        print(
            f"\n上传失败: "
            f"{file_path}\n"
            f"错误: {e}"
        )


# ============================================================
# Reset Statistics
# ============================================================

def reset_statistics():
    global uploaded_bytes
    global total_bytes
    global uploaded_files
    global skipped_files
    global failed_files

    uploaded_bytes = 0
    total_bytes = 0

    uploaded_files = 0
    skipped_files = 0
    failed_files = 0


# ============================================================
# Upload
# ============================================================

def upload_to_server(
        local_path,
        remote_path,
        hostname,
        username,
        password=None,
        port=22,
        max_workers=16
):
    global total_bytes

    # --------------------------------------------------------
    # Reset statistics
    # --------------------------------------------------------

    reset_statistics()

    # --------------------------------------------------------
    # Normalize paths
    # --------------------------------------------------------

    local_path = Path(local_path).resolve()

    # --------------------------------------------------------
    # Scan files
    # --------------------------------------------------------

    upload_files = get_all_upload_files(
        local_path
    )

    total_bytes = sum(
        file_path.stat().st_size
        for file_path in upload_files
    )

    # --------------------------------------------------------
    # Print information
    # --------------------------------------------------------

    print(
        f"待上传文件数量: "
        f"{len(upload_files)}"
    )

    print(
        f"总大小: "
        f"{sizeof_fmt(total_bytes)}"
    )

    print(
        f"线程数: "
        f"{max_workers}"
    )

    # --------------------------------------------------------
    # No files
    # --------------------------------------------------------

    if not upload_files:
        print("没有需要上传的文件")

        return

    # --------------------------------------------------------
    # Thread Pool
    # --------------------------------------------------------

    with ThreadPoolExecutor(
            max_workers=max_workers
    ) as executor:

        futures = [
            executor.submit(
                upload_one_file,
                file_path,
                local_path,
                remote_path,
                hostname,
                username,
                password,
                port,
            )
            for file_path in upload_files
        ]

        for future in as_completed(futures):
            # upload_one_file internally handles
            # exceptions, so this is mainly used
            # to ensure Future has completed.
            future.result()

    # --------------------------------------------------------
    # Close connection of current thread
    #
    # Note:
    # worker threads may already have exited,
    # so ThreadPoolExecutor handles their cleanup.
    # --------------------------------------------------------

    print("\n\n上传完成")

    print(
        f"实际上传文件: "
        f"{uploaded_files}"
    )

    print(
        f"跳过文件: "
        f"{skipped_files}"
    )

    print(
        f"失败文件: "
        f"{failed_files}"
    )

    print(
        f"实际上传大小: "
        f"{sizeof_fmt(uploaded_bytes)}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Password
    # --------------------------------------------------------
    #
    # Recommended:
    #
    # export UPLOAD_PASSWORD='your_password'
    #
    # --------------------------------------------------------

    password = os.getenv(
        "UPLOAD_PASSWORD"
    )

    if not password:
        raise RuntimeError(
            "请设置环境变量 UPLOAD_PASSWORD"
        )

    # --------------------------------------------------------
    # Person detection dataset
    # --------------------------------------------------------

    local_images_path = (
        "/data/database/"
        "AITotal_SegmentDatabase/"
        "personDatabaseSegment/"
        "date_20260826/"
        "images/train"
    )

    local_labels_path = (
        "/data/database/"
        "AITotal_SegmentDatabase/"
        "personDatabaseSegment/"
        "date_20260826/"
        "labels/train"
    )

    remote_images_path = (
        "/home/robot-server/data/"
        "AITotal_SegmentDatabase/"
        "personDatabaseSegment/"
        "images/train"
    )

    remote_labels_path = (
        "/home/robot-server/data/"
        "AITotal_SegmentDatabase/"
        "personDatabaseSegment/"
        "labels/train"
    )

    hostname = "172.16.50.229"

    username = "robot-server"

    # --------------------------------------------------------
    # Upload images
    # --------------------------------------------------------

    upload_to_server(
        local_path=local_images_path,
        remote_path=remote_images_path,
        hostname=hostname,
        username=username,
        password=password,
        max_workers=8,
    )

    # --------------------------------------------------------
    # Upload labels
    # --------------------------------------------------------

    upload_to_server(
        local_path=local_labels_path,
        remote_path=remote_labels_path,
        hostname=hostname,
        username=username,
        password=password,
        max_workers=8,
    )
