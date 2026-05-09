# Real-Time Object Detection and Tracking System

A robust, real-time object detection and tracking system designed for both live webcam feeds and prerecorded video files. This project leverages state-of-the-art deep learning architectures and custom tracking algorithms to provide accurate bounding boxes, continuous object identity tracking, and comprehensive live analytics including zone intrusions and activity heatmaps.

---

## 🌟 Features

- **Real-Time Detection & Classification**: Identifies and classifies objects (living vs. non-living) seamlessly using YOLOv8.
- **Centroid Object Tracking**: Maintains unique IDs across frames utilizing a custom Euclidean distance centroid tracker.
- **Interactive UI & Analytics Dashboards**: Includes a Streamlit web application providing live statistics, charts, and a temporal summary of object movement.
- **Activity Heatmaps**: Spatial analysis of frequent movement zones, overlaid in real-time.
- **ROI Zone Intrusions**: Custom, drawable polygonal zones to monitor boundary crossings.
- **Offline Voice Alerts**: Background audio announcements for new object appearances and zone intrusions using `pyttsx3`.
- **Multithreaded Frame Capture**: Optimized video reading via daemon threads to prevent I/O blocking and maintain high FPS.
- **Session Logging**: Automatically saves tracked object metrics into CSV files for post-analysis.

---

## 🛠️ Tech Stack

- **Core Algorithm**: Ultralytics YOLOv8 (Anchor-free Object Detection)
- **Computer Vision**: OpenCV (`cv2`) for frame manipulation, drawing, and UI
- **Tracking Algorithm**: Custom Centroid Tracker (Euclidean Distance Mapping)
- **Web Interface**: Streamlit, Plotly, Pandas
- **Audio Output**: `pyttsx3`
- **Configuration**: PyYAML

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/object_detection_tracking.git
   cd object_detection_tracking
   ```

2. **Set up a Python Virtual Environment (Recommended):**
   ```bash
   python -m venv yolo_env
   # On Windows:
   yolo_env\Scripts\activate
   # On Linux/Mac:
   source yolo_env/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

The system offers two different interfaces: a native OpenCV window app (`main.py`) and an interactive web dashboard (`app.py`).

### 1. Native OpenCV Application (`main.py`)
This mode is best for high-performance processing and using keyboard shortcuts for active tracking management.

**Run on live webcam:**
*(By default, `allowed_source` in `settings.yaml` is set to `0`)*
```bash
python main.py
```

### 2. Streamlit Web Dashboard (`app.py`)
This mode launches a responsive web app with live metrics, donut charts, and an interactive interface.

**Launch Streamlit:**
```bash
streamlit run app.py
```
*Then open `http://localhost:8501` in your browser.* You can select either your webcam or upload an MP4 video file directly through the sidebar.

---

## ⌨️ Keyboard Controls (for `main.py`)

When running the native OpenCV app, use the following keys while focused on the video window:

- `Q`: Quit the application
- `S`: Save a screenshot to `output/captures/`
- `R`: Start / Stop recording a video clip to `output/clips/`
- `Z`: Enter Zone Mode (Left-click to draw points, Right-click to close polygon, Middle-click to clear)
- `+` / `=`: Increase frame skip (improves display FPS by running YOLO less frequently)
- `-`: Decrease frame skip
- `T`: Toggle trajectory trails
- `H`: Toggle heatmap visibility
- `C`: Clear heatmap data
- `L`: Log stats to terminal
- `V`: Toggle voice alerts on/off

---

## 🖼️ Screenshots

*(Placeholder for screenshots of bounding boxes and tracking IDs)*
> **Tip:** Add your screenshots to an `assets/` folder and link them here using `![Demo](assets/demo.png)`.

---

## 📂 Project Structure

```text
object_detection_tracking/
│
├── core/
│   ├── __init__.py
│   ├── camera_stream.py      # Threaded video capture
│   ├── config.py             # YAML Settings loader
│   ├── detection_logger.py   # CSV Logging system
│   ├── detector.py           # YOLOv8 inference wrapper
│   ├── heatmap.py            # Spatial activity mapping
│   ├── tracker.py            # Centroid tracking algorithm
│   ├── utils.py              # HUD and drawing utilities
│   ├── voice_alert.py        # Background TTS announcements
│   └── zone_manager.py       # ROI polygon drawing and logic
│
├── models/
│   └── yolov8n.pt            # Downloaded YOLOv8 weights
│
├── output/                   # Directory for logs, clips, and captures
├── app.py                    # Streamlit Dashboard Entry Point
├── main.py                   # OpenCV Native App Entry Point
├── requirements.txt          # Pinned Dependencies
└── settings.yaml             # Global Configuration File
```

---

## 🧠 AIML Concepts Covered

This project practically demonstrates several foundational concepts in Artificial Intelligence and Machine Learning (Computer Vision):

1. **Convolutional Neural Networks (CNNs)**: 
   YOLOv8 utilizes a deep CNN backbone (modified CSPDarknet) to extract hierarchical spatial features from video frames.
2. **Anchor-Free Object Detection**: 
   Unlike previous models, YOLOv8 directly predicts object centers and bounding box dimensions without relying on predefined anchor boxes, improving generalization.
3. **Bounding Box Regression**: 
   The model regresses specific coordinates `(x_min, y_min, x_max, y_max)` directly from the feature map to precisely localize objects.
4. **Non-Maximum Suppression (NMS)**: 
   A post-processing filtering algorithm that resolves overlapping predictions. It keeps only the bounding box with the highest confidence score for a single object.
5. **Euclidean Distance & Centroid Tracking**: 
   A tracking algorithm that calculates the Euclidean distance `sqrt((x2 - x1)^2 + (y2 - y1)^2)` between object centroids in consecutive frames to accurately assign and maintain tracking IDs over time.
