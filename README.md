# Real-Time Object Detection and Tracking System

A complete real-time object detection and tracking system that works on both webcam feed and video files. It features clean bounding boxes, class labels, confidence scores, and unique tracking IDs per object.

## How it Works
This project uses **YOLOv8** (You Only Look Once) to detect objects in each frame with high accuracy and speed. We take those bounding box predictions and feed them into a custom **Centroid Tracker**. The tracker calculates the center point of each bounding box and associates objects across consecutive frames by finding the closest match using Euclidean distance.

## Installation

1. Clone or download this project.
2. (Optional) Create a Python virtual environment.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**Run on webcam:**
```bash
python main.py --source 0
```

**Run on a video file:**
```bash
python main.py --source video.mp4
```

**Save output to a video file:**
```bash
python main.py --source 0 --save
```

## Screenshots

*(Placeholder for screenshots of bounding boxes and tracking IDs)*
