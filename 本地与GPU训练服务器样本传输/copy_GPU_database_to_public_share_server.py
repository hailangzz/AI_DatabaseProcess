import paramiko

hostname = "172.16.50.229",
username = "robot-server",
password = "black@box",


def remote_copy_to_share(
        hostname,
        username,
        password,
        source,
        target
):
    ssh = paramiko.SSHClient()

    ssh.set_missing_host_key_policy(
        paramiko.AutoAddPolicy()
    )

    ssh.connect(
        hostname=hostname,
        username=username,
        password=password,
        look_for_keys=False,
        allow_agent=False
    )

    cmd = f"""
    rsync -avh --progress \
    "{source}" \
    "{target}"
    """

    print("执行远程命令:")
    print(cmd)

    stdin, stdout, stderr = ssh.exec_command(cmd)

    print(stdout.read().decode())

    err = stderr.read().decode()

    if err:
        print(err)

    ssh.close()


if __name__ == "__main__":
    remote_copy_to_share(hostname="172.16.50.229", username="robot-server", password="black@box",

                         source=
                         "/home/robot-server/data/AITotal_SegmentDatabase/wireDatabaseSegment",

                         target=
                         "/mnt/share/算法领域/数据集/AI数据集/AISampleDataset/AITotal_SegmentDatabase/"
                         )
