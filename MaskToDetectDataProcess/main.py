from utils import read_database as rd


def main():
    read_database = rd.ReadDatabase("ElectricWiresDataset", "test")
    read_database.get_data_file_name_info()
    read_database.deal_image_masks_picture_data()  # mask标注 -> yolo标注
    read_database.chech_mask_to_detect_effective()

if __name__ == "__main__":
    main()
