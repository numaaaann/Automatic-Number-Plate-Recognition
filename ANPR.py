# import cv2
# from ultralytics import YOLO

# # ---------------------------------------------------------------
# # Load the models:
# #   - vehicle_model: a YOLO model (e.g., YOLO11n) that detects many classes,
# #                    from which we filter only vehicles.
# #   - license_model: your custom-trained YOLO model for license plates.
# # ---------------------------------------------------------------
# vehicle_model = YOLO("yolo11n.pt")         # Pre-trained model (e.g., on COCO)
# license_model = YOLO(r"C:\Users\umrut\OneDrive\Desktop\ANPR\output\custom_train\weights\best.pt")  # Your fine-tuned license plate detector

# # ---------------------------------------------------------------
# # Confidence thresholds (adjust as needed)
# # ---------------------------------------------------------------
# vehicle_conf_threshold = 0.6
# license_conf_threshold = 0.5

# # ---------------------------------------------------------------
# # Define the custom objects to filter vehicles.
# # For example, if using the COCO-trained YOLO model, the following IDs apply:
# # "car": 2, "motorcycle": 3, "bus": 5, "truck": 7.
# # ---------------------------------------------------------------
# custom_objects = {
#     "car": 2,
#     "motorcycle": 3,
#     "bus": 5,
#     "truck": 7,
# }

# # ---------------------------------------------------------------
# # Helper function to scale frames/images while maintaining aspect ratio.
# # ---------------------------------------------------------------
# def scale_frame(frame, max_dimension=800):
#     height, width = frame.shape[:2]
#     if max(height, width) > max_dimension:
#         scaling_factor = max_dimension / max(height, width)
#         new_width = int(width * scaling_factor)
#         new_height = int(height * scaling_factor)
#         return cv2.resize(frame, (new_width, new_height))
#     return frame

# # ---------------------------------------------------------------
# # Helper function to draw a label with a filled background.
# # ---------------------------------------------------------------
# def draw_label(img, text, x, y, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, thickness=2,
#                text_color=(255, 255, 255), bg_color=(0, 0, 0)):
#     # Get text size.
#     (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
#     # Coordinates for the background rectangle.
#     # We'll draw it slightly above the bounding box.
#     rect_x1 = x
#     rect_y1 = max(y - text_height - baseline - 4, 0)
#     rect_x2 = x + text_width + 4
#     rect_y2 = y
#     # Draw the rectangle with a filled background.
#     cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, cv2.FILLED)
#     # Put the text on top.
#     cv2.putText(img, text, (x + 2, y - 2), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

# # ---------------------------------------------------------------
# # Process an image:
# # 1. Read the image.
# # 2. Run the vehicle detector and filter for custom vehicle classes.
# # 3. For each vehicle detection, crop the ROI and run the license plate detector.
# # 4. Draw bounding boxes, labels with confidence scores, and a clear background for text.
# # 5. Scale the output before display.
# # ---------------------------------------------------------------
# def process_image(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         print("Error: Could not load image.")
#         return

#     # Copy the image for annotation.
#     annotated_image = image.copy()

#     # Run the vehicle detector on the full image.
#     vehicle_results = vehicle_model(image, conf=vehicle_conf_threshold)

#     # Loop over vehicle detections.
#     for res in vehicle_results:
#         for box in res.boxes:
#             class_id = int(box.cls[0])
#             # Filter: only process detection if its class ID is one of our custom vehicle classes.
#             if class_id not in custom_objects.values():
#                 continue

#             # Get bounding box coordinates.
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             # Get the confidence score.
#             vehicle_conf = float(box.conf[0])
#             # Create label text with confidence.
#             label = [name for name, cid in custom_objects.items() if cid == class_id]
#             label_text = f"{label[0] if label else 'Vehicle'}: {vehicle_conf:.2f}"

#             # Draw vehicle bounding box (blue).
#             cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             # Draw the label with a filled background.
#             draw_label(annotated_image, label_text, x1, y1, font_scale=0.7, thickness=2,
#                        text_color=(255, 255, 255), bg_color=(0, 0, 255))

#             # Crop the detected vehicle region.
#             vehicle_roi = image[y1:y2, x1:x2]
#             if vehicle_roi.size == 0:
#                 continue

#             # Run the license plate detector on the vehicle ROI.
#             license_results = license_model(vehicle_roi, conf=license_conf_threshold)
#             for lres in license_results:
#                 for lbox in lres.boxes:
#                     lx1, ly1, lx2, ly2 = map(int, lbox.xyxy[0])
#                     # Get the license plate confidence.
#                     lp_conf = float(lbox.conf[0])
#                     # Create license plate label text.
#                     lp_label_text = f"License Plate: {lp_conf:.2f}"
#                     # Convert ROI coordinates to original image coordinates.
#                     gx1 = x1 + lx1
#                     gy1 = y1 + ly1
#                     gx2 = x1 + lx2
#                     gy2 = y1 + ly2
#                     # Draw license plate bounding box (green).
#                     cv2.rectangle(annotated_image, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
#                     # Draw the license plate label.
#                     draw_label(annotated_image, lp_label_text, gx1, gy1,
#                                font_scale=0.7, thickness=2, text_color=(255, 255, 255), bg_color=(0, 128, 0))

#     # Scale the final annotated image before displaying.
#     annotated_image = scale_frame(annotated_image, max_dimension=800)
#     cv2.namedWindow("Detection Result", cv2.WINDOW_NORMAL)
#     cv2.imshow("Detection Result", annotated_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # ---------------------------------------------------------------
# # Process a video stream (or webcam):
# # The logic is similar to image processing but applied on each frame.
# # ---------------------------------------------------------------
# def process_video(video_source):
#     cap = cv2.VideoCapture(video_source)
#     if not cap.isOpened():
#         print("Error: Could not open video source.")
#         return

#     cv2.namedWindow("Detection Result", cv2.WINDOW_NORMAL)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break  # End of video or stream.

#         annotated_frame = frame.copy()

#         # Run vehicle detection on the frame.
#         vehicle_results = vehicle_model(frame, conf=vehicle_conf_threshold)
#         for res in vehicle_results:
#             for box in res.boxes:
#                 class_id = int(box.cls[0])
#                 if class_id not in custom_objects.values():
#                     continue

#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 vehicle_conf = float(box.conf[0])
#                 label = [name for name, cid in custom_objects.items() if cid == class_id]
#                 label_text = f"{label[0] if label else 'Vehicle'}: {vehicle_conf:.2f}"
#                 cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                 draw_label(annotated_frame, label_text, x1, y1,
#                            font_scale=0.7, thickness=2, text_color=(255, 255, 255), bg_color=(0, 0, 255))

#                 # Crop the vehicle region.
#                 vehicle_roi = frame[y1:y2, x1:x2]
#                 if vehicle_roi.size == 0:
#                     continue

#                 # Run license plate detection on the ROI.
#                 license_results = license_model(vehicle_roi, conf=license_conf_threshold)
#                 for lres in license_results:
#                     for lbox in lres.boxes:
#                         lx1, ly1, lx2, ly2 = map(int, lbox.xyxy[0])
#                         lp_conf = float(lbox.conf[0])
#                         lp_label_text = f"License Plate: {lp_conf:.2f}"
#                         gx1 = x1 + lx1
#                         gy1 = y1 + ly1
#                         gx2 = x1 + lx2
#                         gy2 = y1 + ly2
#                         cv2.rectangle(annotated_frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
#                         draw_label(annotated_frame, lp_label_text, gx1, gy1,
#                                    font_scale=0.7, thickness=2, text_color=(255, 255, 255), bg_color=(0, 128, 0))

#         # Scale the annotated frame before display.
#         annotated_frame = scale_frame(annotated_frame, max_dimension=800)
#         cv2.imshow("Detection Result", annotated_frame)
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         # Exit if the window is closed manually.
#         if cv2.getWindowProperty("Detection Result", cv2.WND_PROP_VISIBLE) < 1:
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # ---------------------------------------------------------------
# # Main function:
# # Change the 'source' variable to an image file path, video file path, or "webcam" for live feed.
# # ---------------------------------------------------------------
# def main():
#     source = "sample2.mp4"  # Replace with your image or video file, or use "webcam"
    
#     if isinstance(source, str):
#         if source.lower().endswith(('.png', '.jpg', '.jpeg')):
#             process_image(source)
#         elif source.lower().endswith(('.mp4', '.avi', '.mov')):
#             process_video(source)
#         elif source.lower() == "webcam":
#             process_video(0)
#         else:
#             print("Unsupported file type.")
#     else:
#         print("Invalid source type.")

# if __name__ == "__main__":
#     main()



# import cv2
# from ultralytics import YOLO

# # Load YOLO models
# vehicle_model = YOLO("yolo11n.pt")  # Custom-trained vehicle model
# license_model = YOLO(r"C:\Users\umrut\OneDrive\Desktop\ANPR\output\custom_train\weights\best.pt")  # Custom-trained license plate model

# # Custom object class IDs (modify according to your dataset)
# custom_objects = {
#     "car": 2,
#     "motorcycle": 3,
#     "bus": 5,
#     "truck": 7,
# }

# # Scaling function (for better display)
# def scale_frame(frame, max_dimension=800):
#     height, width = frame.shape[:2]
#     if max(height, width) > max_dimension:
#         scaling_factor = max_dimension / max(height, width)
#         frame = cv2.resize(frame, (int(width * scaling_factor), int(height * scaling_factor)))
#     return frame

# # ✅ **Process Image**
# def process_image(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         print("Error: Could not load image.")
#         return

#     # 🔹 Step 1: Detect Vehicles
#     vehicle_results = vehicle_model(image)

#     for vehicle_result in vehicle_results:
#         detected_boxes = []
        
#         for i, box in enumerate(vehicle_result.boxes):
#             class_id = int(box.cls[0])  # Get class ID of detected object

#             # 🔹 Filter for only custom objects (cars, trucks, buses, motorcycles)
#             if class_id in custom_objects.values():
#                 detected_boxes.append(box)

#         # Update vehicle_result to contain only filtered objects
#         vehicle_result.boxes = detected_boxes

#         # Get annotated vehicle image
#         annotated_img = vehicle_result.plot()

#         # 🔹 Step 2: Detect License Plates inside detected vehicles
#         for box in detected_boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             vehicle_roi = image[y1:y2, x1:x2]

#             if vehicle_roi.size == 0:
#                 continue  # Skip if region is empty

#             # Detect license plates within the vehicle ROI
#             license_results = license_model(vehicle_roi)
#             for license_result in license_results:
#                 for lbox in license_result.boxes:
#                     lx1, ly1, lx2, ly2 = map(int, lbox.xyxy[0])

#                     # Convert license plate coordinates back to full image scale
#                     gx1, gy1, gx2, gy2 = x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2

#                     # Draw license plate box using OpenCV (since YOLO doesn't re-annotate cropped regions)
#                     cv2.rectangle(annotated_img, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
#                     cv2.putText(annotated_img, "License Plate", (gx1, gy1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#         # Scale the final annotated image for display
#         annotated_img = scale_frame(annotated_img)

#         cv2.imshow("Detection Result", annotated_img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

# # ✅ **Process Video**
# def process_video(video_source):
#     cap = cv2.VideoCapture(video_source)
#     if not cap.isOpened():
#         print("Error: Could not open video source.")
#         return

#     cv2.namedWindow("Detection Result", cv2.WINDOW_NORMAL)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # 🔹 Step 1: Detect Vehicles
#         vehicle_results = vehicle_model(frame)

#         for vehicle_result in vehicle_results:
#             detected_boxes = []

#             for i, box in enumerate(vehicle_result.boxes):
#                 class_id = int(box.cls[0])  # Get class ID of detected object

#                 # 🔹 Filter for only custom objects (cars, trucks, buses, motorcycles)
#                 if class_id in custom_objects.values():
#                     detected_boxes.append(box)

#             # Update vehicle_result to contain only filtered objects
#             vehicle_result.boxes = detected_boxes

#             # Get annotated vehicle frame
#             annotated_frame = vehicle_result.plot()

#             # 🔹 Step 2: Detect License Plates inside detected vehicles
#             for box in detected_boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 vehicle_roi = frame[y1:y2, x1:x2]

#                 if vehicle_roi.size == 0:
#                     continue  # Skip if region is empty

#                 # Detect license plates within the vehicle ROI
#                 license_results = license_model(vehicle_roi)
#                 for license_result in license_results:
#                     for lbox in license_result.boxes:
#                         lx1, ly1, lx2, ly2 = map(int, lbox.xyxy[0])

#                         # Convert license plate coordinates back to full image scale
#                         gx1, gy1, gx2, gy2 = x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2

#                         # Draw license plate box using OpenCV
#                         cv2.rectangle(annotated_frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
#                         cv2.putText(annotated_frame, "License Plate", (gx1, gy1 - 10),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#             # Scale the final annotated frame for display
#             annotated_frame = scale_frame(annotated_frame)

#             cv2.imshow("Detection Result", annotated_frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break  # Press 'q' to exit

#     cap.release()
#     cv2.destroyAllWindows()

# # Run the function
# process_image("sample_img.jpg")  # Replace with image path
# # process_video("sample_video.mp4")  # Replace with video path


import cv2
from ultralytics import YOLO

# Load YOLO models
vehicle_model = YOLO("yolo11n.pt")  
license_model = YOLO(r"./output/custom_train/weights/best.pt")  # Custom-trained license plate model

# Custom object class IDs (modify according to your dataset)
custom_objects = {
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7,
}

# Scaling function (for better display)
def scale_frame(frame, max_dimension=800):
    height, width = frame.shape[:2]
    if max(height, width) > max_dimension:
        scaling_factor = max_dimension / max(height, width)
        frame = cv2.resize(frame, (int(width * scaling_factor), int(height * scaling_factor)))
    return frame

# Process Image
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return

    #  Step 1: Detect Vehicles
    vehicle_results = vehicle_model(image)

    for vehicle_result in vehicle_results:
        detected_boxes = []
        
        for i, box in enumerate(vehicle_result.boxes):
            class_id = int(box.cls[0])  # Get class ID of detected object

            #  Filter for only custom objects (cars, trucks, buses, motorcycles)
            if class_id in custom_objects.values():
                detected_boxes.append(box)

        # Update vehicle_result to contain only filtered objects
        vehicle_result.boxes = detected_boxes

        # Get annotated vehicle image
        annotated_img = vehicle_result.plot()

        #  Step 2: Detect License Plates inside detected vehicles
        for box in detected_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vehicle_roi = image[y1:y2, x1:x2]

            if vehicle_roi.size == 0:
                continue  # Skip if region is empty

            # Detect license plates within the vehicle ROI
            license_results = license_model(vehicle_roi)
            for license_result in license_results:
                for lbox in license_result.boxes:
                    lx1, ly1, lx2, ly2 = map(int, lbox.xyxy[0])
                    confidence = float(lbox.conf[0])  # Extract confidence score

                    # Convert license plate coordinates back to full image scale
                    gx1, gy1, gx2, gy2 = x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2

                    # Draw license plate box using OpenCV
                    cv2.rectangle(annotated_img, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)

                    # Add confidence score to the annotation
                    label = f"License Plate ({confidence:.2f})"
                    cv2.putText(annotated_img, label, (gx1, gy1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Scale the final annotated image for display
        annotated_img = scale_frame(annotated_img)

        cv2.imshow("Detection Result", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Process Video
def process_video(video_source):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    cv2.namedWindow("Detection Result", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Detect Vehicles
        vehicle_results = vehicle_model(frame)

        for vehicle_result in vehicle_results:
            detected_boxes = []

            for i, box in enumerate(vehicle_result.boxes):
                class_id = int(box.cls[0])  # Get class ID of detected object

                # Filter for only custom objects (cars, trucks, buses, motorcycles)
                if class_id in custom_objects.values():
                    detected_boxes.append(box)

            # Update vehicle_result to contain only filtered objects
            vehicle_result.boxes = detected_boxes

            # Get annotated vehicle frame
            annotated_frame = vehicle_result.plot()

            # Step 2: Detect License Plates inside detected vehicles
            for box in detected_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicle_roi = frame[y1:y2, x1:x2]

                if vehicle_roi.size == 0:
                    continue  # Skip if region is empty

                # Detect license plates within the vehicle ROI
                license_results = license_model(vehicle_roi)
                for license_result in license_results:
                    for lbox in license_result.boxes:
                        lx1, ly1, lx2, ly2 = map(int, lbox.xyxy[0])
                        confidence = float(lbox.conf[0])  # Extract confidence score

                        # Convert license plate coordinates back to full image scale
                        gx1, gy1, gx2, gy2 = x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2

                        # Draw license plate box using OpenCV
                        cv2.rectangle(annotated_frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)

                        # Add confidence score to the annotation
                        label = f"License Plate ({confidence:.2f})"
                        cv2.putText(annotated_frame, label, (gx1, gy1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Scale the final annotated frame for display
            annotated_frame = scale_frame(annotated_frame)

            cv2.imshow("Detection Result", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Press 'q' to exit

    cap.release()
    cv2.destroyAllWindows()

# def process_video(video_source):
#     cap = cv2.VideoCapture(video_source)
#     if not cap.isOpened():
#         print("Error: Could not open video source.")
#         return

#     cv2.namedWindow("Detection Result", cv2.WINDOW_NORMAL)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.resize(frame, (640, 480))  # Reduce frame size for faster processing

#         vehicle_results = vehicle_model(frame)

#         detected_boxes = []
#         for vehicle_result in vehicle_results:
#             for box in vehicle_result.boxes:
#                 class_id = int(box.cls[0])  
#                 if class_id in custom_objects.values():
#                     detected_boxes.append(box)

#             vehicle_result.boxes = detected_boxes
#             annotated_frame = vehicle_result.plot()

#             for box in detected_boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 vehicle_roi = frame[y1:y2, x1:x2]

#                 if vehicle_roi.size == 0:
#                     continue

#                 license_results = license_model(vehicle_roi)
#                 for license_result in license_results:
#                     for lbox in license_result.boxes:
#                         lx1, ly1, lx2, ly2 = map(int, lbox.xyxy[0])
#                         confidence = float(lbox.conf[0])

#                         gx1, gy1, gx2, gy2 = x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2

#                         cv2.rectangle(annotated_frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
#                         label = f"License Plate ({confidence:.2f})"
#                         cv2.putText(annotated_frame, label, (gx1, gy1 - 10),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#             cv2.imshow("Detection Result", annotated_frame)

#         if cv2.getWindowProperty("Detection Result", cv2.WND_PROP_VISIBLE) < 1:
#             break  # Exit loop if window is closed manually

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break  # Exit loop when 'q' is pressed

#     cap.release()
#     cv2.destroyAllWindows()
#     cv2.waitKey(1)  # Force closure of any lingering windows


def main():
    source = "sample_license.jpg"  # Replace with your image or video file, or use "webcam"
    
    if isinstance(source, str):
        if source.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_image(source)
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
