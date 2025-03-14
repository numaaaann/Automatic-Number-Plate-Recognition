# Automatic Number Plate Recognition (ANPR)

## Overview
This project implements an **Automatic Number Plate Recognition (ANPR)** system using **YOLO-based object detection** and **OCR (Optical Character Recognition)** for extracting license plate numbers from vehicles in a video stream. The system detects vehicles, identifies license plates, and extracts text using **EasyOCR** and **PaddleOCR**, allowing users to choose the best OCR model for their use case.

## Features
- **Vehicle Detection:** Uses YOLO models to detect vehicles such as cars, buses, and trucks.
- **License Plate Detection:** Custom-trained YOLO model for detecting license plates.
- **OCR for Character Recognition:**
  - **EasyOCR**: Works well for standard English characters.
  - **PaddleOCR**: More robust for mixed or non-standard characters.
- **Object Tracking:** Utilizes the **SORT (Simple Online and Realtime Tracker)** algorithm to track vehicles and plates across frames.
- **Visualization:** Displays results in real-time with bounding boxes and extracted text.
- **Customizable Thresholds:** Users can modify confidence thresholds for detections.

## Project Structure
```
ANPR-Project/
│── models/
│   ├── yolo11n.pt  (Pre-trained vehicle detection model)
│   ├── best.pt      (Custom license plate detection model)
│── data/
│   ├── sample_video.mp4 (Test video file)
│── scripts/
│   ├── sort.py      (SORT Tracker)
│── output/
│   ├── processed_video.mp4 (Saved output video)
│── anpr.py          (Main script for running ANPR)
│── requirements.txt (List of dependencies)
│── README.md        (Project documentation)
```


