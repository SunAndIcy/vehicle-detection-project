from utils import load_model, process_image

# Load the YOLOv8 model (cache it in memory to avoid reloading)
model = load_model('/Users/handongdong/pythonProjects/vehicle-detection-project/src/yolov8_model/detect/train4/weights/best.pt')

image_path = '/Users/handongdong/Downloads/models/91727163909_.pic.jpg'  # Replace with your test image path
detections, plate_text = process_image(model, image_path)

for detection in detections:
    print(f"Detection: {detection}")

if plate_text:
    print(f"Detected license plate: {plate_text}")
else:
    print("No license plate detected")
