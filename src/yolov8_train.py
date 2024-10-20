from ultralytics import YOLO

# Load model (lightweight YOLOv8n or YOLOv8s)
model = YOLO('yolov8n.pt')  # You can replace this with 'yolov8s.pt' for experimentation

# Start training
model.train(
    data='/Users/handongdong/pythonProjects/vehicle-detection-project/config.yaml',  # Use absolute path
    epochs=50,           # Number of training epochs
    imgsz=512,           # Image input size (can be set to 416 or 512)
    batch=8,             # Batch size, set according to memory. Recommended 8 or smaller
    device='cpu'         # Use GPU, if no GPU, set to 'cpu'
)
