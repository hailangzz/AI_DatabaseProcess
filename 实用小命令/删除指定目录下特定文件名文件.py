import os
import argparse


def find_files_with_keyword(directory, keyword):
    """
    统计目录下包含关键字的文件数量
    :param directory: 目标目录
    :param keyword: 文件名包含的关键字
    :return: 文件数量
    """
    files = []
    for name in os.listdir(directory):
        if keyword in name:
            files.append(os.path.join(directory, name))
    return files


def delete_files(file_list):
    """
    删除文件
    """
    deleted = 0
    for path in file_list:
        if os.path.exists(path):
            os.remove(path)
            deleted += 1
    return deleted


# 默认路径
DEFAULT_DIR = "/home/chenkejing/database/AITotal_ProjectDatabase/handDatabaseProgrem/images/train"
DEFAULT_KEYWORD = "augment_sample_"


if __name__ == "__main__":
    DEFAULT_delete = False
    # DEFAULT_delete = True
    parser = argparse.ArgumentParser(
        description="Count or delete files containing a specific keyword in a directory"
    )

    parser.add_argument(
        "--dir",
        type=str,
        default=DEFAULT_DIR,
        help=f"Target directory (default: {DEFAULT_DIR})"
    )

    parser.add_argument(
        "--keyword",
        type=str,
        default=DEFAULT_KEYWORD,
        help="Keyword to match in filenames"
    )

    parser.add_argument(
        "--delete",
        type=bool,
        default=DEFAULT_delete,
        help="Delete matched files (True/False), default=False"
    )

    args = parser.parse_args()

    target_dir = args.dir
    keyword = args.keyword

    if not os.path.exists(target_dir):
        print(f"Error: directory '{target_dir}' does not exist!")
        exit()

    matched_files = find_files_with_keyword(target_dir, keyword)

    print(f"Found {len(matched_files)} files containing '{keyword}'")

    if args.delete:
        deleted_num = delete_files(matched_files)
        print(f"Deleted {deleted_num} files.")