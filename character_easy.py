import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
import re

# Load YOLO models
vehicle_model = YOLO("yolo11n.pt")  # Vehicle detection model
license_model = YOLO(r"./output/custom_train/weights/best.pt")  # Custom-trained license plate model

# Initialize EasyOCR reader
ocr_reader = ocr_reader = easyocr.Reader(['en']) 

# Confidence thresholds
vehicle_confidence_threshold = 0.6
license_confidence_threshold = 0.7

# Expand detected license plate bounding box
def expand_bbox(x1, y1, x2, y2, img_shape, margin=20):
    h, w = img_shape[:2]
    x1 = max(0, x1 - margin)  # Expand left boundary, but ensure it’s ≥ 0
    y1 = max(0, y1 - margin)  # Expand top boundary, but ensure it’s ≥ 0
    x2 = min(w, x2 + margin)  # Expand right boundary, but ensure it’s ≤ image width    
    y2 = min(h, y2 + margin)  # Expand bottom boundary, but ensure it’s ≤ image height  
    return x1, y1, x2, y2

# Image Preprocessing for OCR (better contrast and noise removal)
def preprocess_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Smooth the image to reduce noise
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Binary image for better contrast
    return binary_image

# Function to clean OCR results
def clean_ocr_result(text):
    cleaned_text = re.sub(r'[^A-Za-z0-9]', '', text)  # Keep only alphanumeric characters
    return cleaned_text

# Scale function for better display
def scale_frame(frame, max_dimension=1000):
    height, width = frame.shape[:2] 
    if max(height, width) > max_dimension:
        scaling_factor = max_dimension / max(height, width)
        frame = cv2.resize(frame, (int(width * scaling_factor), int(height * scaling_factor)))
    return frame

# Function to enhance text readability
def add_text_with_background(image, text, position, font_scale=1.2, font=cv2.FONT_HERSHEY_SIMPLEX, font_color=(255, 255, 255), thickness=2, margin=5):
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(image, (x - margin, y - margin - h), (x + w + margin, y + margin), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), font, font_scale, font_color, thickness, lineType=cv2.LINE_AA)

# Process Image
def process_image(image):
    # Step 1: Detect Vehicles
    vehicle_results = vehicle_model(image)
    for vehicle_result in vehicle_results:
        detected_boxes = [box for box in vehicle_result.boxes if float(box.conf[0]) >= vehicle_confidence_threshold]
    
        # Step 2: Draw bounding boxes for detected vehicles
        for box in detected_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = vehicle_result.names[class_id]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 4)
            add_text_with_background(image, label, (x1, y1 - 5), font_scale=1.5, font_color=(0, 255, 255))

            vehicle_roi = image[y1:y2, x1:x2]

            # Detect license plates within the vehicle ROI
            license_results = license_model(vehicle_roi)
            for license_result in license_results:
                for lbox in license_result.boxes:
                    lx1, ly1, lx2, ly2 = map(int, lbox.xyxy[0])
                    confidence = float(lbox.conf[0])

                    if confidence >= license_confidence_threshold:
                        gx1, gy1, gx2, gy2 = expand_bbox(x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2, image.shape)

                        license_plate_roi = image[gy1:gy2, gx1:gx2]
                        license_plate_roi = cv2.resize(license_plate_roi, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                        license_plate_roi = preprocess_plate(license_plate_roi)

                        ocr_results = ocr_reader.readtext(license_plate_roi, detail=0, text_threshold=0.5, low_text=0.3)

                        if ocr_results:
                            raw_plate_number = ocr_results[0]
                            plate_number = clean_ocr_result(raw_plate_number)
                            plate_text = f"{plate_number} (Confidence: {confidence:.2f})"
                            add_text_with_background(image, plate_text, (gx1, gy1 - 7), font_scale=2, font_color=(0, 255, 0))

                        cv2.rectangle(image, (gx1, gy1), (gx2, gy2), (0, 255, 0), 4)

    return image

# Process Video
def process_video(video_source):
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_image(frame)

        cv2.imshow('Vehicle and License Plate Detection', scale_frame(processed_frame))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main Function
def main():
    source = r"C:\Users\ASUS\Downloads\ANPR - Copy (1)\ANPR - Copy\samples\local111.jpg"
    if isinstance(source, str):
        if source.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(source)
            if image is not None:
                processed_image = process_image(image)
                cv2.imshow("Detection Result", scale_frame(processed_image))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Error: Could not load image.")
        elif source.lower().endswith(('.mp4', '.avi', '.mov')):
            process_video(source)
        elif source.lower() == "webcam":
            process_video(0)
        else:
            print("Unsupported file type.")
    else:
        print("Invalid source type.")

if __name__ == "__main__":
    main()
