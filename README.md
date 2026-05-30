# Object detection and tracking system

Real-time living/non-living object detection powered by YOLOv8 | Built for AIML internship

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

> A comprehensive real-time computer vision system that detects, classifies, and tracks objects as either living or non-living using advanced neural networks and spatial tracking algorithms.

## Demo

<img src="demo.gif" alt="Object Detection Demo" width="100%">

*Replace with actual recording*

| Living classes (red) | Non-living classes (gray) |
| :--- | :--- |
| person | car |
| bird | chair |
| cat | traffic light |
| dog | cell phone |
| horse | laptop |

## Features

**Detection and Tracking**
- [x] YOLOv8 real-time object detection on webcam or video file
- [x] Living vs Non-living classification — red bounding boxes for living, gray for non-living
- [x] Custom centroid-based object tracker with unique ID per object
- [x] Region of Interest (ROI) zone detection — draw zones by clicking, get alerts when objects enter

**Visualization**
- [x] Object trajectory trails — fading colored path behind each tracked object
- [x] Detection heatmap overlay — shows where objects appear most frequently, toggle with H key
- [x] Streamlit web dashboard with sidebar controls, live metrics, and Plotly charts
- [x] Three live Plotly charts — detection count over time (line chart), living/non-living ratio (donut chart), top 5 classes (bar chart)
- [x] Semi-transparent HUD overlay showing FPS, session timer, object counts, detection latency

**Usability and Performance**
- [x] Threaded video capture using CameraStream class for lag-free frame reading
- [x] Frame skip optimization — YOLO runs every N frames for higher display FPS
- [x] Voice alerts using pyttsx3 offline text-to-speech with per-class cooldown
- [x] Screenshot saver with S key and highlight clip recorder with R key
- [x] Detection CSV logger — saves every detection with timestamp, class, confidence, and coordinates
- [x] settings.yaml config file — all parameters editable without touching Python code
- [x] Auto-save screenshot when simultaneous detection count exceeds threshold

## AIML concepts covered

This project demonstrates practical applications of several fundamental machine learning and computer vision principles.

1. **Convolutional Neural Networks (CNN)** — YOLOv8 backbone extracts spatial features from each frame
2. **Anchor-free object detection** — YOLOv8 predicts bounding boxes without predefined anchor templates
3. **Non-Maximum Suppression (NMS)** — removes duplicate overlapping detections keeping highest confidence box
4. **Bounding box regression** — model predicts x1 y1 x2 y2 coordinates for each detected object
5. **Centroid tracking** — assigns IDs by matching object centroids across frames using Euclidean distance
6. **Euclidean distance** — straight-line distance formula used to associate new detections with existing tracks
7. **Gaussian kernel** — 2D bell-curve kernel used to spread detection points into heatmap blobs
8. **Temporal decay** — heatmap values multiplied by decay factor each frame so older detections fade out
9. **Zone intersection detection** — spatial zone boundary tested using cv2.pointPolygonTest
10. **Real-time inference pipeline** — frame capture to detection to tracking to visualization in under 50ms per frame
11. **Transfer learning** — using YOLOv8 pretrained on COCO without retraining, applied to living/non-living task
12. **Semantic classification** — post-processing model outputs into human categories (living vs non-living)

> These map directly to core AIML curriculum topics, showcasing the bridge between theoretical concepts and applied engineering.

## Tech stack

| Library | Version | Purpose |
| :--- | :--- | :--- |
| Python | 3.10 | Core programming language for the entire backend and logic |
| ultralytics | 8.0.200 | Provides the YOLOv8 model architecture and inference engine |
| opencv-python | 4.8.1 | Handles image matrix operations, drawing, and video streams |
| streamlit | 1.28.0 | Renders the interactive web dashboard and frontend UI |
| plotly | 5.17.0 | Generates the dynamic, live-updating analytics charts |
| numpy | 1.26.1 | Performs rapid mathematical calculations and array manipulations |
| pyttsx3 | 2.90 | Generates offline text-to-speech audio for real-time alerts |
| PyYAML | 6.0.1 | Parses the external configuration file into Python dictionaries |
| pandas | 2.1.2 | Structures the session detection data for CSV file logging |
| Pillow | 10.1.0 | Assists with advanced image handling and frame conversions |

## Project structure

```text
object_detection_tracking/
├── app.py                  (Streamlit web dashboard)
├── main.py                 (OpenCV terminal mode entry point)
├── settings.yaml           (all configurable parameters)
├── requirements.txt
├── README.md
├── .gitignore
├── core/                   (Core backend modules)
│   ├── __init__.py
│   ├── camera_stream.py    (Threaded video capture wrapper)
│   ├── config.py           (settings.yaml configuration loader)
│   ├── detection_logger.py (CSV session logging class)
│   ├── detector.py         (YOLOv8 wrapper)
│   ├── heatmap.py          (Spatial frequency density accumulator)
│   ├── tracker.py          (Centroid object tracker)
│   ├── utils.py            (HUD panel & visualization utilities)
│   ├── voice_alert.py      (Threaded text-to-speech announcements)
│   └── zone_manager.py     (Polygonal ROI zone management)
├── captures/               (auto-created — saved screenshots)
├── clips/                  (auto-created — saved recordings)
├── logs/                   (auto-created — CSV detection logs)
└── screenshots/            (saved presentation screenshots)
```

The application is built with a highly modular architecture that separates the neural network inference from the visualization and UI components. The `detector.py` file acts as a standalone wrapper for YOLOv8, ensuring that any future model upgrades won't break the downstream logic. System configuration is fully abstracted into `settings.yaml`, allowing parameters to be tuned dynamically at startup. By using `camera_stream.py` to move frame reading into a background thread, the system guarantees that the main application loop remains unblocked and performs at the highest possible framerate.

## Installation

### Prerequisites

- Python 3.8 to 3.11 installed on your system
- pip package manager enabled
- A connected USB webcam or a local video file (MP4 format)
- 8GB RAM minimum for smooth real-time processing

1. Clone the repository to your local machine:
```bash
git clone https://github.com/yourusername/object_detection_tracking.git
cd object_detection_tracking
```

2. Create a virtual environment:

For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

For Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install all required dependencies:
```bash
pip install -r requirements.txt
```

4. Verify the installation by triggering the config parser:
```bash
python config.py
```

> Note: The required YOLOv8 nano model weights file (`yolov8n.pt`) is not included in the repository by default. It downloads automatically from the Ultralytics servers upon your first run.

> Note: No dedicated GPU is required to run this project. The YOLOv8 nano architecture is highly optimized and will run effectively on standard consumer CPUs.

## How to run

### Method 1 — Streamlit dashboard (recommended)

```bash
streamlit run app.py
```

![Streamlit Dashboard UI Placeholder](https://via.placeholder.com/800x450.png?text=Streamlit+Dashboard+UI)

> Tip: We highly recommend using the Streamlit dashboard as it provides the most comprehensive visual experience, featuring live telemetry and interactive graphing capabilities.

### Method 2 — Terminal webcam

```bash
python main.py --source 0
```

### Method 3 — Terminal video file

```bash
python main.py --source path/to/video.mp4
```

### Method 4 — Terminal with saved output

```bash
python main.py --source 0 --save
```

## Keyboard controls

| Key | Action | Notes |
| :--- | :--- | :--- |
| Q | Quit the application | Safely closes all threads and releases the camera resource |
| S | Save screenshot to captures/ folder | Generates a timestamped image file immediately |
| R | Start or stop recording clip to clips/ folder | Toggles the background video writer state |
| H | Toggle heatmap overlay on/off | Blends the density map over the current frame |
| T | Toggle trajectory trails on/off | Displays the historical path of all active IDs |
| V | Toggle voice alerts on/off | Enables or disables pyttsx3 speech synthesis output |
| Z | Activate zone drawing mode | Press Z, then click on the video window to draw an ROI |
| L | Print current session log stats to terminal | Dumps current metrics to the console stdout |
| Plus key | Increase frame skip | Lowers CPU usage by inferencing less frequently |
| Minus key | Decrease frame skip | Increases tracking accuracy at the cost of CPU cycles |
| Left click | Add point to zone polygon in zone mode | Only registers clicks when Z mode is active |
| Right click | Close and save current zone polygon | Connects your final point back to the origin |
| Middle click | Clear all zones | Instantly deletes all active monitoring areas |

## Configuration

The behavior of the tracking system can be customized via the `settings.yaml` configuration file without modifying any Python code.

```yaml
# settings.yaml - Master Configuration File

detection:
  # Minimum confidence score (0.0 to 1.0) required to register an object
  confidence_threshold: 0.45
  
  # Number of frames to skip between YOLO inferences for performance optimization
  skip_frames: 2
  
  # Path to the pretrained model weights file
  model_path: "yolov8n.pt"
  
  # Allowed source types (0 for webcam, or explicit file paths)
  allowed_source: "0"

tracking:
  # Maximum consecutive frames an object can be lost before its ID is deregistered
  max_disappeared: 30
  
  # Maximum pixel distance between centroids to still be considered the same object
  distance_threshold: 50.0

display:
  # BGR color code for living objects (Red: Blue=0, Green=0, Red=255)
  living_color_bgr: [0, 0, 255]
  
  # BGR color code for non-living objects (Gray: Blue=128, Green=128, Red=128)
  nonliving_color_bgr: [128, 128, 128]
  
  # BGR color code for the text labels
  label_text_color: [255, 255, 255]
  
  # Boolean flag to toggle the top-left FPS counter display
  fps_display: true
  
  # Boolean flag to toggle the default state of trajectory trails
  show_trails: true
  
  # Number of historical points to keep for each object's trajectory tail
  trail_length: 20

recording:
  # Number of simultaneous objects required to trigger an automatic screenshot
  auto_save_threshold: 5
  
  # Boolean flag to automatically begin recording video upon application launch
  record_on_start: false
  
  # Desired frames per second for the output MP4 recording files
  output_fps: 30.0

alerts:
  # Boolean flag to enable or disable the text-to-speech engine globally
  voice_enabled: true
  
  # Minimum seconds to wait before repeating a voice alert for the same class
  voice_cooldown_seconds: 10
  
  # Boolean flag to enable visual and audio warnings for ROI zone intrusions
  zone_alert_enabled: true
```

| Parameter | Default | Description | Valid Range |
| :--- | :--- | :--- | :--- |
| confidence_threshold | 0.45 | Minimum confidence for detection | 0.0 to 1.0 |
| skip_frames | 2 | Inference intervals for performance | 0 to 10 |
| model_path | yolov8n.pt | Pretrained weights file location | Valid file path |
| allowed_source | 0 | Authorized video input stream | integer or path |
| max_disappeared | 30 | Frames to wait before dropping ID | 1 to 100 |
| distance_threshold | 50.0 | Pixel radius for ID matching | 10.0 to 200.0 |
| living_color_bgr | [0, 0, 255] | Bounding box color for living items | BGR array |
| nonliving_color_bgr | [128, 128, 128]| Bounding box color for non-living | BGR array |
| label_text_color | [255, 255, 255]| Color of the classification text | BGR array |
| fps_display | true | Show framerate on the HUD | true / false |
| show_trails | true | Render historical motion paths | true / false |
| trail_length | 20 | Number of coordinates in trail | 5 to 100 |
| auto_save_threshold | 5 | Crowd count to trigger screenshot | 1 to 50 |
| record_on_start | false | Auto-capture video on boot | true / false |
| output_fps | 30.0 | Framerate of the saved clip | 15.0 to 60.0 |
| voice_enabled | true | Master switch for TTS output | true / false |
| voice_cooldown_seconds| 10 | Delay between audio notifications | 1 to 300 |
| zone_alert_enabled | true | Trigger events for ROI boundary | true / false |

## Screenshots

<p align="center">
  <img src="screenshots/dashboard.jpg" alt="Streamlit Dashboard" width="48%">
  <img src="screenshots/terminal_mode.jpg" alt="Terminal Mode" width="48%">
</p>
<p align="center">
  <img src="screenshots/heatmap.jpg" alt="Heatmap View" width="48%">
  <img src="screenshots/zones.jpg" alt="Zone Alert" width="48%">
</p>

*Note: Add actual screenshots to the screenshots/ folder after testing.*

```bash
mkdir screenshots
```

## Recording a demo

1. Download and install OBS Studio on your local machine.
2. Add a new "Window Capture" source pointing to the active Streamlit or OpenCV window.
3. Record exactly 90 seconds of footage, demonstrating object tracking, zone drawing, and heatmap toggling.
4. Upload the resulting video file to ezgif.com for web optimization.
5. Convert the video, download the final file, and save it as `demo.gif` in the root folder of this repository.

## How it works

```text
[Video Input] 
     │
     ▼
[Frame Capture] ──(Threaded Buffer)──┐
                                     │
     ┌───────────────────────────────┘
     ▼
[YOLOv8 Detection] ──▶ Extracts Boxes, Classes, Confidence
     │
     ▼
[Living/Non-living Classification] ──▶ Maps 80 COCO classes to binary groups
     │
     ▼
[Centroid Tracking] ──▶ Calculates distance, assigns persistent IDs
     │
     ▼
[Zone Check] ──▶ Tests pointPolygonTest for ROI boundary intersections
     │
     ▼
[Visualization] ──▶ Overlays bounding boxes, trails, HUD, and heatmaps
     │
     ▼
[Output] ──▶ Renders to OpenCV Window or Streamlit Web Dashboard
```

The system operates as a continuous, linear pipeline optimized for speed. First, the camera stream captures frames in a dedicated background thread to prevent I/O blocking. These frames are passed into the YOLOv8 neural network, which identifies objects and outputs their raw bounding box coordinates and class IDs. A classification layer maps these standard COCO classes into "Living" or "Non-living" categories, determining their color codes. The centroid tracker then calculates the Euclidean distance between new detections and previous frames to maintain persistent object IDs. Finally, the visualization layer paints the tracking data onto the frame and passes the finished image to either the local display or the Streamlit server.

## Future improvements

*   Integrate custom model training to recognize specific hardware components or unique inventory items.
*   Replace the Euclidean distance tracker with DeepSORT integration for better occlusion handling and re-identification.
*   Add support for network camera streaming via RTSP protocols for remote security deployments.
*   Develop a companion mobile app to receive real-time push notifications of zone intrusions.
*   Containerize the application with Docker for seamless cloud deployment and orchestration.
*   Implement an automated email or SMS zone alert system using third-party API webhooks.

## License and author

This project is licensed under the MIT License - see the LICENSE file for details.

**Author:** Srijan Sahu  
**Institution:** Intern Candidate  
**Year:** 2026  

Built as part of the AIML internship program, May 2026.  
[GitHub](https://github.com/srijansahu) | [LinkedIn](https://linkedin.com/in/srijan-sahu)
