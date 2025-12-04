import os
import utils.util as util
import cv2


class ReadDatabase:

    def __init__(self, database_name,data_part=None, origin_path=r"/home/chenkejing/database/carpetDatabase"):
        """

        :param database_name: 数据库名称
        :param data_part: 数据库组成train、test``
        :param origin_path: 数据库存储根目录
        """
        self.database_path = database_name
        self.data_part = data_part
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

    def get_data_file_name_info(self):
        if self.data_part is None:
            part_database_direct_name_list = os.listdir(self.database_path)
            for database_part in part_database_direct_name_list:
                self.image_path = os.path.join(self.database_path, database_part, self.image_direct_name)
                self.image_name_list = util.read_name_list(self.image_path)

                self.masks_path = os.path.join(self.database_path, database_part, self.mask_direct_name)
                self.masks_name_list = util.read_name_list(self.masks_path)
        else:
            self.image_path = os.path.join(self.database_path, self.data_part, self.image_direct_name)
            self.image_name_list = util.read_name_list(self.image_path)

            self.masks_path = os.path.join(self.database_path, self.data_part, self.mask_direct_name)
            self.masks_name_list = util.read_name_list(self.masks_path)


    #将mask标注，转为detect标注
    def deal_image_masks_picture_data(self, deal_type="MaskeToDetect"):

        if deal_type == "MaskeToDetect":
            util.mark_to_detect(self.masks_path)

    #检查detect标注的有效性
    def chech_mask_to_detect_effective(self):
        util.use_yolo_label_plot_box(self.image_path)
def test():
    read_database = ReadDatabase("SUTD_NEW.v3-sutd.coco","test")
    read_database.get_data_file_name_info()
    read_database.deal_image_masks_picture_data()
    read_database.chech_mask_to_detect_effective()


if __name__ == "__main__":
    test()
