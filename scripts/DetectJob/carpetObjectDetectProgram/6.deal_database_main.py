import os
import utils.util as util

#对训练集中，样本标签信息，进行统一替换（解决不同样本集，同一目标class_id不一致的问题）
# ===== 示例调用 =====
if __name__ == "__main__":
    util.replace_yolo_class_id("/home/chenkejing/database/AITotal_ProjectDatabase/carpetDatabaseProgrem/labels/train", 0)
