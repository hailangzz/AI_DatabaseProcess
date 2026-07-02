"""
远程服务器样本清洗：
    数据集里面有 images 和 labels 两个文件夹，里面的文件名是对应的，但是有些图片没有对应的txt，有些txt没有对应的图片。请写一个python脚本，删除不对应的文件，并输出统计信息。
思路如下：

    SSH连接服务器
    SFTP读取 images 和 labels
    建立图片basename集合
    建立txt basename集合
    求差集
    删除：
    图片存在，txt不存在 → 删除图片
    txt存在，图片不存在 → 删除txt
    输出统计信息
"""

import os

import paramiko
from paramiko import SSHClient

IMAGE_SUFFIX = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def get_remote_files(sftp, remote_dir, suffix_set):
    """
    获取远程目录下指定后缀文件
    返回:
        {
            basename: remote_path
        }
    """
    result = {}

    for filename in sftp.listdir(remote_dir):

        ext = os.path.splitext(filename)[1].lower()

        if ext not in suffix_set:
            continue

        basename = os.path.splitext(filename)[0]

        result[basename] = os.path.join(remote_dir, filename)

    return result


def clean_remote_dataset(
        image_remote_path,
        label_remote_path,
        hostname,
        username,
        password=None,
        port=22,
):
    global DRY_RUN
    ssh = SSHClient()

    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    print(f"连接服务器 {hostname}...")

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

        print("服务器连接成功")

        sftp = ssh.open_sftp()

        image_dict = get_remote_files(
            sftp,
            image_remote_path,
            IMAGE_SUFFIX,
        )

        label_dict = get_remote_files(
            sftp,
            label_remote_path,
            {".txt"},
        )

        image_set = set(image_dict.keys())

        label_set = set(label_dict.keys())

        ############################
        # 图片有，标签没有
        ############################

        remove_images = sorted(image_set - label_set)

        ############################
        # 标签有，图片没有
        ############################

        remove_labels = sorted(label_set - image_set)

        print()

        print("========== 清洗结果 ==========")

        print(f"图片总数 : {len(image_set)}")
        print(f"标签总数 : {len(label_set)}")

        print()

        print(f"删除图片数量 : {len(remove_images)}")
        print(f"删除标签数量 : {len(remove_labels)}")

        print()

        ############################
        # 删除图片
        ############################

        for name in remove_images:

            remote_file = image_dict[name]

            if DRY_RUN:
                print(f"[DRY RUN] 删除图片: {remote_file}")
            else:
                print(f"删除图片: {remote_file}")
                sftp.remove(remote_file)

        ############################
        # 删除标签
        ############################

        for name in remove_labels:

            remote_file = label_dict[name]

            if DRY_RUN:
                print(f"[DRY RUN] 删除标签: {remote_file}")
            else:
                print(f"删除标签: {remote_file}")
                sftp.remove(remote_file)

        print()

        print("数据清洗完成")

        sftp.close()

        ssh.close()

    except Exception as e:

        print(e)


if __name__ == "__main__":
    # 设置为 True 进行测试，不会实际删除文件，设置为 False 才会删除文件
    # DRY_RUN = True  # 测试，不会删除文件
    DRY_RUN = False  # 正式删除文件

    image_remote_path = "/home/robot-server/data/AITotal_SegmentDatabase/plasticbagDatabaseSegment/images/train"
    label_remote_path = "/home/robot-server/data/AITotal_SegmentDatabase/plasticbagDatabaseSegment/labels/train"

    hostname = "172.16.50.229"
    username = "robot-server"
    password = "black@box"

    clean_remote_dataset(
        image_remote_path,
        label_remote_path,
        hostname,
        username,
        password,
    )
