import torch
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
import os
import ssl

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Custom collate_fn to handle variable-length annotations
def custom_collate_fn(batch):
    images = [item[0] for item in batch]  # Extract images
    targets = [item[1] for item in batch]  # Extract annotations
    return images, targets

# Convert COCO format bounding boxes [x_min, y_min, width, height] to [x_min, y_min, x_max, y_max]
def convert_bbox_format(boxes):
    new_boxes = []
    for box in boxes:
        x_min, y_min, width, height = box
        x_max = x_min + width
        y_max = y_min + height
        if width > 0 and height > 0:  # Ensure the width and height are positive
            new_boxes.append([x_min, y_min, x_max, y_max])
    return torch.tensor(new_boxes, dtype=torch.float32)

# Validate the model
def validate_model(val_loader, model):
    model.eval()

    with torch.no_grad():
        for images, targets in val_loader:
            images = list(img.to(device) for img in images)

            # Convert target format similarly
            new_targets = []
            for t in targets:
                boxes = convert_bbox_format([obj['bbox'] for obj in t])  # Convert bounding box format
                labels = torch.tensor([obj['category_id'] for obj in t], dtype=torch.int64).to(device)
                new_targets.append({"boxes": boxes.to(device), "labels": labels})

            # Forward pass to get predictions
            predictions = model(images)

            # Output predictions
            print(f"Validation predictions: {predictions}")

# Main training process
def train_model(num_epochs, train_loader, val_loader, model, optimizer):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = list(img.to(device) for img in images)

            # Convert COCO format targets to the format expected by SSD
            new_targets = []
            for t in targets:
                boxes = convert_bbox_format([obj['bbox'] for obj in t])  # Convert bounding box format
                labels = torch.tensor([obj['category_id'] for obj in t], dtype=torch.int64).to(device)
                new_targets.append({"boxes": boxes.to(device), "labels": labels})

            # Calculate loss
            loss_dict = model(images, new_targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')

        # Validate the model
        validate_model(val_loader, model)

# Data preparation
def prepare_data(batch_size, root, annFile):
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((300, 300)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CocoDetection(root=root, annFile=annFile, transform=transform)

    # Split dataset into training (80%) and validation (20%) sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Data loaders, using custom collate_fn
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

    return train_loader, val_loader

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == "__main__":
    num_classes = 3  # Classes are 'car', 'license_plate', and background
    batch_size = 16
    num_epochs = 50
    learning_rate = 1e-4

    # Load the pre-trained SSD model
    weights = SSD300_VGG16_Weights.DEFAULT
    model = ssd300_vgg16(weights=weights)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Data paths
    root = '/Users/handongdong/Downloads/kaikeba/coco/train2017'
    annFile = '/Users/handongdong/Downloads/kaikeba/coco/annotations/instances_default.json'

    # Prepare data (including training and validation set splitting)
    train_loader, val_loader = prepare_data(batch_size, root, annFile)

    # Create directory to save model checkpoints
    os.makedirs('ssd_model', exist_ok=True)

    # Start training
    train_model(num_epochs, train_loader, val_loader, model, optimizer)

    # Save the final model
    torch.save(model.state_dict(), os.path.join('ssd_model', 'ssd_coco_car_license_plate_final.pth'))
