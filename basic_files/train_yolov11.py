from ultralytics import YOLO
# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML

# Train the model
results = model.train(data="viso.yaml", epochs=0, imgsz=640)
# results = model.val()  # evaluate model performance on the validation set
# results = model.export(format='onnx')  # export the model to ONNX format
  
# # Load a model
# model2 = YOLO("yolo11n.yaml")  # build a new model from YAML

# # Train the model
# results2 = model2.train(data="dota.yaml", epochs=150, imgsz=640)
# results2 = model2.val()  # evaluate model performance on the validation set
# results2 = model2.export(format='onnx')  # export the model to ONNX format