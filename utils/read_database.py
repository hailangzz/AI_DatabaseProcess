import os
import utils.util as util
import cv2


class ReadDatabase:

    def __init__(self, database_name, origin_path=r"/home/chenkejing/database"):
        self.database_path = None
        self.image_direct_name = 'imgs'
        self.mask_direct_name = "masks"
        self.get_database_info(database_name, origin_path)

        self.image_path = None
        self.masks_path = None
        self.image_name_list = []
        self.masks_name_list = []

    def get_database_info(self, database_name, origin_path):
        database_info = {"origin_path": origin_path,
                         "database_name": database_name}
        self.database_path = os.path.join(database_info["origin_path"], database_info["database_name"])

    def get_data_file_name_info(self, data_part="test"):
        part_database_direct_name_list = os.listdir(self.database_path)
        for database_part in part_database_direct_name_list:
            self.image_path = os.path.join(self.database_path, database_part, self.image_direct_name)
            self.image_name_list = util.read_name_list(self.image_path)

            self.masks_path = os.path.join(self.database_path, database_part, self.mask_direct_name)
            self.masks_name_list = util.read_name_list(self.masks_path)

    def deal_image_masks_picture_data(self, deal_type="MaskeToDetect"):

        if deal_type == "MaskeToDetect":
            util.mark_to_detect(self.masks_path)

    def chech_mask_to_detect_effective(self):
        util.use_yolo_label_plot_box(self.image_path)
def test():
    read_database = ReadDatabase("ElectricWiresDataset")
    read_database.get_data_file_name_info()
    read_database.deal_image_masks_picture_data()
    read_database.chech_mask_to_detect_effective()


if __name__ == "__main__":
    test()
