# Real-Time Object Detection and Tracking

This directory contains code for a real-time object detection and tracking system that utilizes YOLOv8 for object detection and a centroid-based tracking algorithm.

## Overview

The primary goal of this system is to detect objects in real-time video feeds and maintain their identity across frames. This is particularly useful for applications such as surveillance, traffic monitoring, and robotics.

## Technologies Used
- **YOLOv8**: A state-of-the-art object detection model that provides high accuracy and speed for real-time applications.
- **Centroid Tracking**: A simple yet effective method to track moving objects based on their centroid positions.

## Getting Started

### Prerequisites
- Python 3.x
- Required libraries (see requirements.txt)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Srijan0409/CodeAlpha_AI-Object-detection-and-tracking.git
   ```
2. Navigate to the project directory:
   ```bash
   cd CodeAlpha_AI-Object-detection-and-tracking/object_detection_tracking
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. Run the detection script:
   ```bash
   python detect_and_track.py
   ```
2. Ensure the video source is correctly set in the script (e.g. webcam or video file).

**Key Components:
detector.py - YOLOv8-based object detection module that:

Loads pretrained YOLOv8 model
Detects objects from frames with configurable confidence threshold
Classifies objects as "living" or "non-living" (people, animals vs objects)
Uses CNN backbone and anchor-free detection
tracker.py - Custom centroid-based tracking module that:

Maintains unique IDs for detected objects
Associates detections across frames using Euclidean distance
Handles object registration and deregistration
Tracks up to 80 COCO classes
main.py - Main entry point featuring:

Real-time video processing pipeline
Support for webcam or video file input
Live FPS counter and statistics display
Optional video output saving
Color-coded visualization (Red for living objects, Gray for non-living)
utils.py - Visualization utilities:

Drawing rounded bounding boxes with tracking IDs
FPS calculation and display
Statistics panel showing object counts by class
requirements.txt - Dependencies:

ultralytics==8.0.196 (YOLOv8)
opencv-python==4.8.1.78
numpy==1.24.3**

## Contributing

Contributions are welcome! Please create a pull request for any changes or enhancements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*Date Created: 2026-05-03 09:04:19 (UTC)*
