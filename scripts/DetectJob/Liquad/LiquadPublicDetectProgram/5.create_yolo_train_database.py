import os
import utils.util as util

database_source_path = r"/home/chenkejing/database/AITotal_ProjectDatabase/handDatabaseProgrem"
util.create_director_for_yolo_train_databse(database_source_path)

database_part_type = ["train", "val"]

for part_type in database_part_type:
    src_imgs_director = os.path.join(database_source_path, part_type, "imgs")
    src_labels_director = os.path.join(database_source_path, part_type, "labels")

    des_imgs_director = os.path.join(database_source_path, "images", part_type)
    des_labels_director = os.path.join(database_source_path, "labels", part_type)

    util.copy_yolo_dataset(src_imgs_director, src_labels_director, des_imgs_director, des_labels_director)