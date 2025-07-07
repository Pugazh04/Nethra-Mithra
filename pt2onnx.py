from ultralytics import YOLO

# Load the YOLOv8m model
model = YOLO("yolov8s.pt")

# Export the model to ONNX format
model.export(format="onnx")
