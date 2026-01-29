"""
# 说明：
    在使用cd /home/chenkejing/git_director/SAM_Auto_Label/segment-anything 做标注时，由于偶尔打断，导致embedding 和 images不匹配。此时需要将两个文件夹下数据对齐、
    又或者，先自动embedding一半，开始标注，标注过程中，if 让它继续embedding:
"""

import os
import shutil

class DealEmbeddingImage:

    def __init__(self):
        self.embedding_path="/home/chenkejing/database/HandDetect/EmdoorRealHandImages/embeddings"
        self.images_path="/home/chenkejing/database/HandDetect/EmdoorRealHandImages/images"

        self.embedding_name_list = []
        self.images_name_list = []

        self.ends_string_type_images = ""
        self.unshare_name_list = []
        self.save_unshare_image_path = "/home/chenkejing/database/HandDetect/EmdoorRealHandImages/unshare_images/images"

    def deal_embedding_images(self,):
        self.ends_string_type_images = os.listdir(self.images_path)[0][-4:]
        # self.embedding_name_list = [name[:-4] for name in os.listdir(self.embedding_path)]
        for name in os.listdir(self.embedding_path):

            self.embedding_name_list.append(name[:-4])

        for name in os.listdir(self.images_path):
            self.images_name_list.append(name[:-4])

        print(len(self.embedding_name_list))
        print(len(self.images_name_list))
        self.unshare_name_list = list(set(self.images_name_list) - set(self.embedding_name_list))

        print(self.unshare_name_list)
        # 移动未编码图像：
        # for name in self.unshare_name_list:
        src_image_path = os.path.join(self.images_path,name+self.ends_string_type_images)
        dst_image_path = os.path.join(self.save_unshare_image_path+self.ends_string_type_images)
        print(src_image_path)
        print(dst_image_path)
            # shutil.move(src_image_path, dst_image_path)


deal_embedding_images = DealEmbeddingImage()
deal_embedding_images.deal_embedding_images()