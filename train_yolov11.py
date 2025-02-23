from ultralytics import YOLO

model = YOLO('yolo_test.yaml')

# Train the model
results = model.train(data="viso.yaml", epochs=1, imgsz=640)