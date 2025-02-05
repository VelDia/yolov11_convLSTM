
import ultralytics
ultralytics.checks()
from roboflow import Roboflow
rf = Roboflow(api_key="TGqdgMvcnZrzN0D51LXs")
project = rf.workspace("igor-na").project("dota-v2.0---easy")
version = project.version(7)
dataset = version.download("yolov11")

# from ultralytics import YOLO

# # Load a model
# model = YOLO("yolo11n.yaml")  # build a new model from YAML

# # Train the model
# results = model.train(data="viso.yaml", epochs=100, imgsz=640)
# results = model.val()  # evaluate model performance on the validation set
# results = model.export(format='onnx')  # export the model to ONNX format