from ultralytics import YOLO

# Load the best trained model
model = YOLO('E:\\computerVision\\train3\\weights\\best.pt')

# Run prediction on a test folder
results = model.predict(source='test/images', save=True, imgsz=640, conf=0.25)
