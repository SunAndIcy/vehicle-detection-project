import cv2
import easyocr
import pytesseract
import numpy as np
from ultralytics import YOLO


# Load YOLOv8 model
def load_model(weights_path):
    model = YOLO(weights_path)
    model.conf = 0.6  # Confidence threshold, can be adjusted as needed
    model.imgsz = 640  # Image size
    return model


# Filter license plate recognition results, keeping only characters in the whitelist, and limiting the number of characters to 6
def filter_license_plate_text(plate_text):
    PLATE_WHITELIST = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    filtered_text = ''.join([char for char in plate_text if char in PLATE_WHITELIST])

    # Limit the license plate length to 6 characters
    if len(filtered_text) > 6:
        filtered_text = filtered_text[:6]

    return filtered_text


# Increase brightness
def increase_brightness(plate_img, alpha=1.5, beta=50):
    bright_img = cv2.convertScaleAbs(plate_img, alpha=alpha, beta=beta)
    return bright_img


# Gamma correction
def apply_gamma_correction(plate_img, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected_img = cv2.LUT(plate_img, table)
    return gamma_corrected_img


# Adaptive histogram equalization (CLAHE)
def apply_clahe(plate_img):
    gray_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(gray_img)
    return enhanced_img


# Image processing function for license plate preprocessing (for dark images)
def preprocess_plate_image(plate_img):
    # Step 1: Brightness enhancement
    bright_img = increase_brightness(plate_img, alpha=1.5, beta=50)

    # Step 2: Gamma correction
    gamma_corrected_img = apply_gamma_correction(bright_img, gamma=1.5)

    # Step 3: Local contrast enhancement (CLAHE)
    clahe_img = apply_clahe(gamma_corrected_img)

    # Step 4: Image sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp_img = cv2.filter2D(clahe_img, -1, kernel)

    # Step 5: Binarization using fixed thresholding
    _, binary_img = cv2.threshold(sharp_img, 120, 255, cv2.THRESH_BINARY_INV)

    return binary_img


# Image resizing (to enlarge the license plate image before OCR)
def resize_plate(plate_img, scale_factor=4):
    height, width = plate_img.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized_plate = cv2.resize(plate_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Add some padding, increasing the area around the license plate
    padded_plate = cv2.copyMakeBorder(resized_plate, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    return padded_plate


# Multi-model OCR fusion
def fuse_ocr_results(plate_img, reader):
    # EasyOCR result
    easyocr_result = reader.readtext(plate_img)

    # Tesseract configuration and result
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    tesseract_result = pytesseract.image_to_string(plate_img, config=custom_config).strip()

    # Print OCR recognition results
    print(f"EasyOCR result: {easyocr_result}")
    print(f"Tesseract result: {tesseract_result}")

    # If EasyOCR returns a result with high confidence, prioritize using EasyOCR's result
    if easyocr_result and len(easyocr_result) > 0:
        easyocr_text = easyocr_result[0][-2]
        if easyocr_result[0][-1] > 0.6:  # Confidence greater than 0.6
            print(f"Using EasyOCR result: {easyocr_text}")
            return easyocr_text

    # If the EasyOCR result is not ideal, use the Tesseract result
    print(f"Using Tesseract result: {tesseract_result}")
    return tesseract_result


# Process vehicle and license plate detection
def process_detections(results, img):
    detections = []
    license_plate_text = None
    reader = easyocr.Reader(['en'])  # Use EasyOCR's English language model

    for box in results[0].boxes:
        xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()  # Get the bounding box coordinates
        conf = box.conf[0].cpu().numpy()  # Confidence
        cls = int(box.cls[0].cpu().numpy())  # Class

        detection = {
            "class": cls,
            "confidence": float(conf),
            "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)]
        }

        if cls == 1:  # License plate detected
            # Extract the license plate image
            plate_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
            plate_img = resize_plate(plate_img)  # Dynamically resize the license plate image
            processed_plate_img = preprocess_plate_image(plate_img)

            # Perform OCR recognition
            license_plate_text = fuse_ocr_results(processed_plate_img, reader)
            if license_plate_text:
                license_plate_text = filter_license_plate_text(license_plate_text)

            detection["label"] = "license_plate"
            detection["text"] = license_plate_text

        detections.append(detection)

    return detections, license_plate_text


# Process the image
def process_image(model, image_path):
    img = cv2.imread(image_path)  # Read the image
    results = model(img)  # Perform detection using the YOLO model
    detections, plate_text = process_detections(results, img)  # Process the detection results
    return detections, plate_text
