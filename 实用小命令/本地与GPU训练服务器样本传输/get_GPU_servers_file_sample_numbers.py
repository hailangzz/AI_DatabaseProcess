import paramiko
from paramiko import SSHClient


def exec_count_cmd(ssh, cmd):
    """
    执行统计命令
    """

    stdin, stdout, stderr = ssh.exec_command(cmd)

    result = stdout.read().decode().strip()

    return int(result)


def count_remote_dataset(
    image_remote_path,
    label_remote_path,
    hostname,
    username,
    password=None,
    port=22,
):

    ssh = SSHClient()

    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

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

        # 图像统计命令
        image_cmd = (
            f'find "{image_remote_path}" -type f '
            f'\\( -iname "*.jpg" -o '
            f'-iname "*.jpeg" -o '
            f'-iname "*.png" -o '
            f'-iname "*.bmp" -o '
            f'-iname "*.webp" \\) | wc -l'
        )

        # txt统计命令
        txt_cmd = (
            f'find "{label_remote_path}" -type f '
            f'-iname "*.txt" | wc -l'
        )

        # 获取数量
        image_count = exec_count_cmd(ssh, image_cmd)

        txt_count = exec_count_cmd(ssh, txt_cmd)

        print("\n===== 数据集统计结果 =====")

        print(f"图像路径: {image_remote_path}")
        print(f"图像数量: {image_count}")

        print()

        print(f"标注路径: {label_remote_path}")
        print(f"TXT数量: {txt_count}")

        # 一致性检查
        print()

        if image_count == txt_count:
            print("数据集检查正常：图像数量 与 TXT数量 一致")
        else:
            print("警告：图像数量 与 TXT数量 不一致")

        ssh.close()

        return image_count, txt_count

    except Exception as e:

        print(f"\n连接失败: {e}")

        return None, None


if __name__ == "__main__":

    # image_remote_path = "/home/robot-server/data/AITotal_SegmentDatabase/wireDatabaseSegment_all_database/images/train"
    # label_remote_path = "/home/robot-server/data/AITotal_SegmentDatabase/wireDatabaseSegment_all_database/labels/train"

    image_remote_path = "/home/robot-server/data/AITotal_SegmentDatabase/wireDatabaseSegment/images/train"
    label_remote_path = "/home/robot-server/data/AITotal_SegmentDatabase/wireDatabaseSegment/labels/train"

    hostname = "172.16.50.229"
    username = "robot-server"
    password = "black@box"

    count_remote_dataset(
        image_remote_path,
        label_remote_path,
        hostname,
        username,
        password,
    )