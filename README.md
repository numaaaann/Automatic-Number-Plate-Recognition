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

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed.

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/anpr-project.git
   cd anpr-project
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Required Dependencies
- OpenCV
- NumPy
- Ultralytics YOLO
- EasyOCR
- PaddleOCR
- SORT Tracker

## Usage
### Running the ANPR System
```sh
python anpr.py --video data/sample_video.mp4 --ocr paddle
```

### Arguments
- `--video`: Path to the input video file.
- `--ocr`: Select OCR model (`easy` for EasyOCR, `paddle` for PaddleOCR).

### Output
- Processed video is saved in the **output/** directory.
- Real-time detection is displayed in a window.

## Customization
### Modify Detection Confidence Thresholds
Adjust the confidence thresholds for vehicle and license plate detection in `anpr.py`:
```python
vehicle_confidence_threshold = 0.6
license_confidence_threshold = 0.7
```

### Expanding Bounding Box for OCR
You can fine-tune the margin for license plate cropping:
```python
def expand_bbox(x1, y1, x2, y2, img_shape, margin=10):
    h, w = img_shape[:2]
    return max(0, x1 - margin), max(0, y1 - margin), min(w, x2 + margin), min(h, y2 + margin)
```


## Author
- **Mohammad Numaan**
- GitHub: (https://github.com/numaaaann)

