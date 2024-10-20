import torch
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from PIL import Image, ImageDraw
import torchvision.transforms as transforms

# Category mapping
category_mapping = {1: 'car', 2: 'license_plate'}

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = SSD300_VGG16_Weights.DEFAULT
model = ssd300_vgg16(weights=weights)
model.load_state_dict(torch.load('ssd_model/ssd_coco_car_license_plate_final.pth', map_location=device))
model.to(device)
model.eval()

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Can remove Normalize for testing
])

# Load and process the image
img_path = '/Users/handongdong/Downloads/models/22.jpg'
image = Image.open(img_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)  # Convert to Tensor and add batch dimension

# Inference
with torch.no_grad():
    predictions = model(image_tensor)

# Display prediction results
boxes = predictions[0]['boxes'].cpu()
labels = predictions[0]['labels'].cpu()
scores = predictions[0]['scores'].cpu()

# Set confidence threshold
confidence_threshold = 0.5
selected_indices = scores > confidence_threshold
boxes = boxes[selected_indices]
labels = labels[selected_indices]
scores = scores[selected_indices]

# Output bounding boxes, labels, and confidence scores
for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
    print(f"Prediction {i + 1}:")
    print(f"  Box: {box}")
    print(f"  Label: {category_mapping.get(label.item(), 'Unknown')}")
    print(f"  Score: {score}")

# Visualize results
draw = ImageDraw.Draw(image)
for box, label, score in zip(boxes, labels, scores):
    x_min, y_min, x_max, y_max = box
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
    draw.text((x_min, y_min), f"{category_mapping.get(label.item(), 'Unknown')}: {score:.2f}", fill="red")

# Show the image
image.show()
