import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import easyocr
import re
import os
import imageio

# Load YOLO models
vehicle_model = YOLO("yolo11n.pt") # This loads the pre-trained YOLO model for vehicle detection
license_model = YOLO(r"./output/custom_train/weights/best.pt") # Custom-trained license plate model

# Initialize EasyOCR reader
ocr_reader = easyocr.Reader(['en']) # Initializes the OCR reader for English language

# Class labels dictionary
CLASS_LABELS = {
    0: 'P',    # Person
    1: 'B',    # Bike
    2: 'C',    # Car
    3: 'M',    # Motorcycle
    5: 'B',    # Bus
    7: 'T',    # Truck
    9: 'T.L'   # Traffic Light
}

# Color mapping for each class (in BGR format)
CLASS_COLORS = {
    0: (255, 0, 0),     # Person - Blue
    1: (0, 165, 255),   # Bike - Orange
    2: (0, 0, 255),     # Car - Red
    3: (255, 0, 255),   # Motorcycle - Magenta
    5: (0, 255, 0),     # Bus - Green
    7: (255, 255, 0),   # Truck - Cyan
    9: (128, 0, 128)    # Traffic Light - Purple
}

# Dictionary to store object counts per class
class_counters = {label: 0 for label in CLASS_LABELS.values()}

# Confidence thresholds
vehicle_confidence_threshold = 0.6  # Minimum confidence required to consider a vehicle detection valid
license_confidence_threshold = 0.7  # Minimum confidence required to consider a license plate detection valid

# Expand detected license plate bounding box
def expand_bbox(x1, y1, x2, y2, img_shape, margin=10):
    h, w = img_shape[:2]
    x1 = max(0, x1 - margin)  # Expand left boundary, but ensure it’s ≥ 0
    y1 = max(0, y1 - margin)  # Expand top boundary, but ensure it’s ≥ 0
    x2 = min(w, x2 + margin)  # Expand right boundary, but ensure it’s ≤ image width    
    y2 = min(h, y2 + margin)  # Expand bottom boundary, but ensure it’s ≤ image height  
    return x1, y1, x2, y2


# Image Preprocessing for OCR
def preprocess_plate(image): 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # since ocr models work better w grayscale imgs, therefore we convert 3 channels BGR into single channel/grayscale
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # smooths out the random variables, removing unwanted small details that can interfere w ocr
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  # dynamically adjusts the threshold for diff parts of the img, ensures even in shadows or brighter areas the txt remains visible
    return gray


# Function to clean OCR results
def clean_ocr_result(text):
    cleaned_text = re.sub(r'[^A-Za-z0-9]', '', text)  # Keep only alphanumeric characters
    return cleaned_text 

# Scale function for better display
def scale_frame(frame, max_dimension=800):
    height, width = frame.shape[:2]
    if max(height, width) > max_dimension:
        scaling_factor = max_dimension / max(height, width)
        frame = cv2.resize(frame, (int(width * scaling_factor), int(height * scaling_factor)))
    return frame


# Function to enhance text readability
def add_text_with_background(image, text, position, color=(255, 255, 255), font_scale=0.8, font=cv2.FONT_HERSHEY_SIMPLEX, thickness=2, margin=5):
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness) # measures txt size, calcs w&h of text wrt chosen font, font_scale, thickness
    x, y = position
    cv2.rectangle(image, (x - margin, y - margin - h), (x + w + margin, y + margin), (0, 0, 0), -1) # Background rectangle for better contrast
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness, lineType=cv2.LINE_AA) # draws the text on img at (x,y), line type=cv2.lineaa is used to smooth any altering text

# Process Video
def process_video(video_source): # This initializes video capture from the given video_source.
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Create output filename based on input
    base_name = os.path.splitext(os.path.basename(video_source))[0]
    output_path = f"{base_name}_processed.mp4"
    
    # List to store processed frames
    processed_frames = []
    
    print(f"Processing video...")
    print(f"Output will be saved to: {output_path}")

    tolerance_seconds = 1.5 # How many seconds the tracker should wait when an object disappears
    max_age = int(fps * tolerance_seconds)  # Convert seconds to number of frames to wait

# Initializes the SORT tracker to maintain unique IDs for detected vehicles across frames.
    tracker = Sort(max_age=max_age, min_hits=1, iou_threshold=0.3)


    # Dictionary to store class IDs and original detections for tracked objects
    tracked_classes = {}  # Maps tracked object IDs to their class (e.g., car, bus, truck)
    detection_map = {}    # Temporarily stores detections for tracking

    frame_count = 0  # Counter to track processed frames


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            break

        frame_count += 1  #Increments the frame counter.
        print(f"Processing frame {frame_count}/{int(total_frames)}", end='\r')  #Prints the processing progress dynamically.

        vehicle_results = vehicle_model(frame)  # Detect vehicles in the frame
        detected_boxes = []  # Stores bounding boxes of detected vehicles
        current_detections = []  # Stores detection information for tracking


        for vehicle_result in vehicle_results:
            for box in vehicle_result.boxes:
                confidence = float(box.conf[0])
                if confidence >= vehicle_confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'class_id': class_id,
                        'confidence': confidence
                    }
                    current_detections.append(detection)
                    detected_boxes.append([x1, y1, x2, y2, confidence])
                    
                    if class_id in [2, 5, 7]:  # Car, Bus, Truck
                        vehicle_roi = frame[y1:y2, x1:x2]
                        license_results = license_model(vehicle_roi)
                        
                        for license_result in license_results:
                            for lbox in license_result.boxes:
                                l_confidence = float(lbox.conf[0])
                                if l_confidence >= license_confidence_threshold:
                                    lx1, ly1, lx2, ly2 = map(int, lbox.xyxy[0])
                                    gx1, gy1, gx2, gy2 = expand_bbox(x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2, frame.shape)
                                    
                                    license_plate_roi = frame[gy1:gy2, gx1:gx2]
                                    if license_plate_roi.size > 0:
                                        license_plate_roi = cv2.resize(license_plate_roi, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                                        license_plate_roi = preprocess_plate(license_plate_roi)
                                        
                                        ocr_results = ocr_reader.readtext(license_plate_roi, detail=0, text_threshold=0.6)
                                        
                                        plate_color = CLASS_COLORS.get(class_id, (0, 255, 0))
                                        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), plate_color, 4)
                                        for text in ocr_results:
                                            plate_text = f"{text} ({l_confidence:.2f})"
                                            add_text_with_background(frame, plate_text, (gx1, gy1 - 10), color=plate_color)

        # Update tracker
        detections = np.array(detected_boxes) if detected_boxes else np.empty((0, 5))
        tracked_objects = tracker.update(detections)

        # Update tracked classes
        detection_map.clear()
        for detection in current_detections:
            min_dist = float('inf')
            best_track_id = None
            for track in tracked_objects:
                tx1, ty1, tx2, ty2, track_id = map(int, track)
                dx1, dy1, dx2, dy2 = detection['bbox']
                track_center = ((tx1 + tx2) / 2, (ty1 + ty2) / 2)
                detect_center = ((dx1 + dx2) / 2, (dy1 + dy2) / 2)
                dist = np.sqrt((track_center[0] - detect_center[0]) ** 2 + (track_center[1] - detect_center[1]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_track_id = int(track_id)
            if best_track_id is not None:
                tracked_classes[best_track_id] = detection['class_id']

        # Draw tracked objects
        for track in tracked_objects:
            tx1, ty1, tx2, ty2, track_id = map(int, track)
            track_id = int(track_id)
            if track_id in tracked_classes:
                class_id = tracked_classes[track_id]
                if class_id in CLASS_LABELS:
                    class_label = CLASS_LABELS[class_id]
                    color = CLASS_COLORS.get(class_id, (0, 255, 255))
                    if track_id not in class_counters:
                        if class_label not in class_counters:
                            class_counters[class_label] = 0
                        class_counters[class_label] += 1
                        class_counters[track_id] = class_counters[class_label]
                    cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), color, 3)
                    label_text = f"{class_label}{class_counters[track_id]}"
                    add_text_with_background(frame, label_text, (tx1, ty1 - 10), color=color)

        # Scale frame before saving and displaying
        frame = scale_frame(frame)
        
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frames.append(frame_rgb)
        
        # Display frame
        cv2.imshow("Detection Result", frame)

        if cv2.waitKey(200) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nProcessing complete. Saving video...")
    
    try:
        # Save video using imageio
        imageio.mimsave(output_path, processed_frames, fps=fps)
        print(f"Video saved successfully to: {output_path}")
    except Exception as e:
        print(f"Error saving video: {e}")
        # Try alternative save location
        try:
            alternative_path = os.path.join(os.getcwd(), output_path)
            imageio.mimsave(alternative_path, processed_frames, fps=fps)
            print(f"Video saved successfully to alternative location: {alternative_path}")
        except Exception as e:
            print(f"Failed to save video to alternative location: {e}")

def main():
    source = r"C:\Users\ASUS\Downloads\ANPR - Copy (1)\ANPR - Copy\sample.mp4"
    process_video(source)

if __name__ == "__main__":
    main()