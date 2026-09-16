import os
import shutil


def copy_to_shared_server(src_path, dest_dir):
    """
    将文件或文件夹拷贝到共享服务器的指定目录。

    :param src_path: 要拷贝的本地文件或文件夹路径
    :param dest_dir: 共享服务器的目标目录路径
    """
    # 1. 检查目标共享目录是否存在（防止共享连接断开）
    if not os.path.exists(dest_dir):
        print(f"❌ 错误: 找不到目标共享目录！")
        print(f"请检查共享服务器是否已断开连接，或路径是否正确。")
        print(f"当前尝试访问的路径: {dest_dir}")
        return False

    # 2. 检查源文件/文件夹是否存在
    if not os.path.exists(src_path):
        print(f"❌ 错误: 源路径不存在！ -> {src_path}")
        return False

    try:
        # 获取源文件/文件夹的名称
        base_name = os.path.basename(os.path.normpath(src_path))
        target_path = os.path.join(dest_dir, base_name)

        # 3. 执行拷贝
        if os.path.isdir(src_path):
            print(f"📂 正在拷贝文件夹: {src_path} ...")
            print(f"➡️ 目标位置: {target_path}")
            # 如果目标已存在同名文件夹，shutil.copytree 会报错，这里做个防护
            if os.path.exists(target_path):
                print(f"⚠️ 警告: 目标目录已存在 {base_name}，正在尝试覆盖/合并...")
                # 也可以选择删除旧的：shutil.rmtree(target_path)
            shutil.copytree(src_path, target_path, dirs_exist_ok=True)
        else:
            print(f"📄 正在拷贝文件: {src_path} ...")
            print(f"➡️ 目标位置: {target_path}")
            shutil.copy(src_path, target_path)

        print("✨ 拷贝完成！")
        return True

    except Exception as e:
        print(f"💥 拷贝过程中发生异常: {e}")
        return False


if __name__ == "__main__":
    # ===== 1. 定义你的目标共享目录 =====
    # 建议使用 raw 字符串 (r"...") 避免中文字符或特殊符号转义问题
    DESTINATION_DIR = r"/run/user/1001/gvfs/smb-share:server=172.16.50.250,share=share/算法领域/数据集/AI数据集/AISampleDataset/AITotal_SegmentDatabase"

    # ===== 2. 定义你要拷贝的本地文件或文件夹 =====
    # 请将这里替换为你本地实际想拷贝的路径
    SOURCE_PATH = r"/home/chenkejing/sam2"

    # 执行拷贝
    copy_to_shared_server(SOURCE_PATH, DESTINATION_DIR)
