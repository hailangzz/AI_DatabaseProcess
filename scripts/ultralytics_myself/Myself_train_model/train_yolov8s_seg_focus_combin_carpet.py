from ultralytics import YOLO

# # 训练地毯识别模型
# model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/models/v8/yolov8s_focus_wire.yaml")  # load a pretrained model (recommended for training)
# model.load("/home/chenkejing/PycharmProjects/ultralytics/yolov8s.pt")
# model.load("/home/chenkejing/PycharmProjects/ultralytics/runs/my_carpet_exp/yolov8_focus_v1/weights/last.pt")
# results = model.train(data="carpet_detect.yaml", epochs=100, imgsz=640, device=0, workers=0, resume=True, project="runs/my_carpet_exp", name="yolov8_focus_v1")
#


# # 训练线材检测模型
# model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/models/v8/yolov8s_focus_carpet.yaml")  # load a pretrained model (recommended for training)
# # model.load("/home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_carpet_exp/yolov8_focus_sa_v2/weights/best.pt")
# model.load("/home/chenkejing/PycharmProjects/ultralytics/yolov8s.pt")
# results = model.train(data="carpet_detect.yaml", epochs=300, imgsz=640, device=-1, workers=0, batch=32, project="runs/my_carpet_exp", name="yolov8_focus_v")
#



from ultralytics import YOLO

if __name__ == "__main__":

    # 1️⃣ 加载分割模型结构（seg）
    model = YOLO("/home/chenkejing/git_director/ultralytics/ultralytics/cfg/models/v8/yolov8-seg_focus_combin_carpet.yaml")

    # 2️⃣ 加载预训练权重（非常重要）
    model.load("/home/chenkejing/PycharmProjects/ultralytics/yolov8s-seg.pt")
    # model.load("/home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_carpet_seg_exp/yolov8s_carpet_seg_v1_6/weights/best.pt")

    # 3️⃣ 开始训练
    results = model.train(
        task="segment",                 # ⭐ 必须指定
        data="cross_database_multis-seg.yaml",      # 分割数据集 yaml
        epochs=300,
        imgsz=640,
        batch=14,                       # seg 比 detect 更吃显存
        device=0,                       # -1 = CPU，0 = GPU
        workers=0,

        optimizer="SGD",                # ⭐ 稳定，利于部署
        cos_lr=True,
        amp=False,                      # ⭐ ONNX/RKNN 强烈建议关  /amp 控制的是「训得快不快 vs 稳不稳」

        project="runs/my_carpet_seg_exp",
        name="yolov8s_carpet_seg_v1_",
        # resume=False                    # 控制的是「训不训旧的状态」
        resume=True
    )

    # watch -n 1 nvidia-smi #监控GPU占用信息
