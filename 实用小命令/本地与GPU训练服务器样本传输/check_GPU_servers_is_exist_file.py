import paramiko
from paramiko import SSHClient


def search_remote_file(
        remote_dir,
        target_filename,
        hostname,
        username,
        password=None,
        port=22,
):
    """
    在远程服务器指定目录下搜索文件
    """

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

        # find 搜索命令
        search_cmd = f'find "{remote_dir}" -type f ' f'-name "{target_filename}"'

        print(f"\n执行搜索命令:\n{search_cmd}\n")

        stdin, stdout, stderr = ssh.exec_command(search_cmd)

        result = stdout.read().decode().strip()

        error = stderr.read().decode().strip()

        if error:
            print(f"搜索错误: {error}")
            return None

        if result:
            print("成功！！！\n 找到目标文件:\n")

            file_list = result.split("\n")

            for file_path in file_list:
                print(file_path)

            ssh.close()

            return file_list

        else:
            print("失败！！！\n 未找到目标文件")

            ssh.close()

            return []

    except Exception as e:

        print(f"\n连接失败: {e}")

        return None


if __name__ == "__main__":
    # remote_dir = "/home/robot-server/data/AITotal_SegmentDatabase/wireDatabaseSegment/images/train"
    # remote_dir = "/home/robot-server/data/AITotal_SegmentDatabase/wireDatabaseSegment_all_database/images/train"
    remote_dir = "/home/robot-server/data/AITotal_SegmentDatabase/liquidDatabaseSegment/images/train"
    #
    # # 要搜索的文件名
    # target_filename = "20260522_150800_920_liquid_detect.jpg"

    # remote_dir = "/home/robot-server/data/AITotal_SegmentDatabase/wireDatabaseSegment/images/train"
    # remote_dir = "/home/robot-server/data/AITotal_SegmentDatabase/liquidDatabaseSegment/labels/train"

    # 要搜索的文件名
    target_filename = "20260522_075115_735_liquid_detect.jpg"

    hostname = "172.16.50.229"
    username = "robot-server"
    password = "black@box"

    search_remote_file(
        remote_dir,
        target_filename,
        hostname,
        username,
        password,
    )
