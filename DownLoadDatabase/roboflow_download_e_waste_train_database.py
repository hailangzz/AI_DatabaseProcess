from roboflow import Roboflow

rf = Roboflow(api_key="MbNDxdQ67FV0G7ANCio6")
project = rf.workspace("datasets-qlvxa").project("e-waste-train")
version = project.version(1)
dataset = version.download("yolov5-obb")
