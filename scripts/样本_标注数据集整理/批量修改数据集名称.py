import os

def rename_files(directory, old_str, new_str):
    for filename in os.listdir(directory):
        if old_str in filename:
            new_name = filename.replace(old_str, new_str)

            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)

            os.rename(old_path, new_path)
            print(f"{filename} -> {new_name}")

if __name__ == "__main__":
    dir_path = "/data/database/AITotal_Segment_ValDatabase/public_real_camera_images_0422_carpet_val_batch1"
    rename_files(dir_path, "real_liquid_image_batch5_soy_", "real_carpet_val_image_batch1_")