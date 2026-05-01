"""
Utility functions for drawing detections and statistics on video frames.
"""

import cv2
import time
import numpy as np
from typing import Dict, Tuple, List, Any

# Fixed color map for classes
COLOR_MAP = {
    "person": (255, 100, 100),
    "car": (100, 255, 100),
    "truck": (100, 100, 255),
    "bus": (255, 255, 100),
    "motorcycle": (255, 100, 255),
    "bicycle": (100, 255, 255),
    "dog": (200, 200, 200),
    "cat": (100, 100, 100)
}

def get_color(class_name: str) -> Tuple[int, int, int]:
    """Helper to get a color for a class, default to white if not found."""
    return COLOR_MAP.get(class_name, (255, 255, 255))

def draw_rounded_rect(img, pt1, pt2, color, thickness, r, d):
    """Draws a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Draw straight lines
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)

    # Draw arcs
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

def draw_detections(frame: np.ndarray, tracked_objects: Dict[int, Tuple[int, int, List[int]]], class_names: Dict[int, str], confidences: Dict[int, float]):
    """
    Draws bounding boxes, labels, and tracking IDs on the frame.
    
    Args:
        frame: The OpenCV image/frame.
        tracked_objects: Dict from tracker mapping ID to (cX, cY, bbox).
        class_names: Dict mapping object ID to class name.
        confidences: Dict mapping object ID to confidence score.
    """
    for object_id, info in tracked_objects.items():
        cx, cy, bbox = info
        x1, y1, x2, y2 = bbox
        
        class_name = class_names.get(object_id, "unknown")
        color = get_color(class_name)
        conf = confidences.get(object_id, 0.0)
        
        # Draw smooth rounded corner boxes
        draw_rounded_rect(frame, (x1, y1), (x2, y2), color, 2, 10, 10)
        
        # Label format: "Person #3 | 94%"
        if conf > 0:
            label = f"{class_name.capitalize()} #{object_id} | {int(conf * 100)}%"
        else:
            label = f"{class_name.capitalize()} #{object_id}"
            
        # Filled label background rectangle for readability
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
        
        # Text
        cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def calculate_fps(prev_time: float) -> Tuple[float, float]:
    """
    Calculates current FPS.
    
    Args:
        prev_time: Timestamp of the previous frame.
        
    Returns:
        Tuple[float, float]: (current_fps, current_time)
    """
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0.0
    return fps, current_time

def draw_fps(frame: np.ndarray, fps: float):
    """
    Overlays FPS counter on top-left corner.
    """
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def draw_stats(frame: np.ndarray, object_count_dict: Dict[str, int]):
    """
    Draws a small panel bottom-left showing count per class.
    
    Args:
        frame: OpenCV image/frame.
        object_count_dict: Dictionary mapping class names to counts.
    """
    h, w = frame.shape[:2]
    
    stats_text = " | ".join([f"{k.capitalize()}s: {v}" for k, v in object_count_dict.items() if v > 0])
    if not stats_text:
        stats_text = "No objects detected"
        
    # Draw background panel
    cv2.rectangle(frame, (10, h - 40), (w - 10, h - 10), (0, 0, 0), -1)
    # Draw text
    cv2.putText(frame, stats_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
