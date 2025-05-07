from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

# Train the model using the dataset
model.train(
    data='E:/computerVision/tomatoLeaf/data.yaml',
    epochs=5,
    imgsz=640,
    batch=16
)