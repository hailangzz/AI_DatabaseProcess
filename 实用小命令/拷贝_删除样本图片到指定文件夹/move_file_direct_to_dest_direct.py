import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


def move_single_file(args):
    """
    移动单个文件
    """

    src_file, dst_file, overwrite = args

    try:

        # 创建目标目录
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)

        # 文件存在处理
        if os.path.exists(dst_file):

            if overwrite:
                os.remove(dst_file)

            else:
                return False, "skip"

        shutil.move(src_file, dst_file)

        return True, "move"

    except Exception as e:

        return False, str(e)


def collect_files(src_dir):
    """
    收集所有文件路径
    """

    files = []

    for root, dirs, filenames in os.walk(src_dir):

        for filename in filenames:
            src_file = os.path.join(root, filename)

            files.append(src_file)

    return files


def move_all_files(src_dir, dst_dir, workers=16, overwrite=False):
    """
    多线程移动目录下所有文件

    参数:
        src_dir:
            源目录

        dst_dir:
            目标目录

        workers:
            线程数

        overwrite:
            是否覆盖
    """

    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"source directory not exists: {src_dir}")

    os.makedirs(dst_dir, exist_ok=True)

    # 收集文件
    files = collect_files(src_dir)

    print(f"total files: {len(files)}")

    tasks = []

    for src_file in files:
        relative_path = os.path.relpath(src_file, src_dir)

        dst_file = os.path.join(dst_dir, relative_path)

        tasks.append((src_file, dst_file, overwrite))

    moved = 0
    skipped = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:

        futures = [executor.submit(move_single_file, task) for task in tasks]

        for future in tqdm(
                as_completed(futures), total=len(futures), desc="Moving files"
        ):

            success, status = future.result()

            if success:

                moved += 1

            elif status == "skip":

                skipped += 1

            else:

                failed += 1

    print("\n========== Done ==========")

    print(f"moved   : {moved}")

    print(f"skipped : {skipped}")

    print(f"failed  : {failed}")


if __name__ == "__main__":
    src_dir = "/home/chenkejing/database/No_Target_Example_Dataset/No_Target_database/NO_target_camera_images_0407_batch1_output_empty_only/person_labels"

    dst_dir = "/data/database/AITotal_SegmentDatabase/personDatabaseSegment/labels/train"

    move_all_files(src_dir, dst_dir, workers=32, overwrite=False)
