"""
Utility functions for drawing detections and statistics on video frames.
"""

import cv2
import time
import numpy as np
from typing import Dict, Tuple, List, Any
from config import CFG

LIVING_COLOR = CFG.living_color    # Red   in BGR (OpenCV format)
NONLIVING_COLOR = CFG.nonliving_color # Gray  in BGR

def draw_trails(frame: np.ndarray, tracker, tracked_objects: dict, is_living_dict: dict) -> np.ndarray:
    """
    Draw fading polyline trails behind each tracked object.
    Living objects: red trail. Non-living: gray trail.
    Trail fades from bright (recent) to transparent (oldest point).
    Uses alpha blending per segment for the fade effect.
    """
    for obj_id in tracked_objects:
        trail = tracker.get_trail(obj_id)
        if len(trail) < 2:
            continue
        
        is_living = is_living_dict.get(obj_id, False)
        base_color = LIVING_COLOR if is_living else NONLIVING_COLOR
        
        # Draw each segment with increasing opacity towards recent end
        for i in range(1, len(trail)):
            alpha = i / len(trail)  # 0.0 = oldest, 1.0 = most recent
            thickness = max(1, int(alpha * 3))
            
            # Blend color with black based on alpha for fade effect
            faded_color = tuple(max(0, min(255, int(c * alpha))) for c in base_color)
            
            pt1 = trail[i-1]
            pt2 = trail[i]
            cv2.line(frame, pt1, pt2, faded_color, thickness, cv2.LINE_AA)
        
        # Draw a small dot at the current position (brightest point)
        if trail:
            cv2.circle(frame, trail[-1], 3, base_color, -1)
    
    return frame

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

def draw_detections(frame: np.ndarray, tracked_objects: Dict[int, Tuple[int, int, List[int]]], class_names: Dict[int, str], confidences: Dict[int, float], is_living_dict: Dict[int, bool]):
    """
    Draws bounding boxes, labels, and tracking IDs on the frame.
    
    Args:
        frame: The OpenCV image/frame.
        tracked_objects: Dict from tracker mapping ID to (cX, cY, bbox).
        class_names: Dict mapping object ID to class name.
        confidences: Dict mapping object ID to confidence score.
        is_living_dict: Dict mapping object ID to living boolean.
    """
    for object_id, info in tracked_objects.items():
        cx, cy, bbox = info
        x1, y1, x2, y2 = bbox
        
        class_name = class_names.get(object_id, "unknown")
        is_living = is_living_dict.get(object_id, False)
        
        color = LIVING_COLOR if is_living else NONLIVING_COLOR
        icon = "🔴" if is_living else "⬜"
        
        # Draw smooth rounded corner boxes
        draw_rounded_rect(frame, (x1, y1), (x2, y2), color, 2, 10, 10)
        
        # Label format: "Dog #4 🔴" for living, "Chair #7 ⬜" for non-living
        label = f"{class_name.capitalize()} #{object_id} {icon}"
            
        # Filled label background rectangle for readability
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
        
        # Text
        cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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

def draw_hud_panel(frame: np.ndarray, living_count: int, nonliving_count: int, fps: float, session_seconds: int, detection_time_ms: float = 0, skip_frames: int = 1) -> np.ndarray:
    """
    Draws a semi-transparent black panel in the bottom-left corner showing live stats.
    
    Args:
        frame: OpenCV image/frame.
        living_count: Total count of living objects.
        nonliving_count: Total count of non-living objects.
        fps: Current frames per second.
        session_seconds: Elapsed session time in seconds.
        detection_time_ms: Latency of the last detection pass.
        skip_frames: Current frame skip setting.
        
    Returns:
        Modified OpenCV image/frame.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    x1, y1 = 10, h - 160
    x2, y2 = x1 + 220, y1 + 150
    
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    
    text_x = x1 + 10
    start_y = y1 + 20
    
    cv2.putText(frame, "LIVE STATS", (text_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    
    start_y += 18
    cv2.circle(frame, (text_x + 5, start_y - 4), 5, (0, 0, 220), -1)
    cv2.putText(frame, f"Living: {living_count}", (text_x + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    start_y += 18
    cv2.circle(frame, (text_x + 5, start_y - 4), 5, (160, 160, 160), -1)
    cv2.putText(frame, f"Non-living: {nonliving_count}", (text_x + 15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    
    start_y += 18
    cv2.putText(frame, f"FPS: {fps:.1f}", (text_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    start_y += 18
    cv2.putText(frame, f"Session: {session_seconds}s", (text_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    
    start_y += 18
    cv2.putText(frame, f"Det: {detection_time_ms:.0f}ms", (text_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    start_y += 18
    cv2.putText(frame, f"Skip: 1/{skip_frames}", (text_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    return frame

def draw_fps_counter(frame: np.ndarray, fps: float) -> np.ndarray:
    """
    Draws an FPS counter in the top-right corner with a background rectangle.
    
    Args:
        frame: OpenCV image/frame.
        fps: Current frames per second.
        
    Returns:
        Modified OpenCV image/frame.
    """
    if fps >= 20:
        color = (0, 200, 0)
    elif fps >= 10:
        color = (0, 200, 200)
    else:
        color = (0, 0, 200)
        
    if fps == 0:
        text = "FPS --"
    else:
        text = f"FPS {fps:.0f}"
        
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    h, w = frame.shape[:2]
    x = w - text_width - 10
    y = 30
    
    cv2.rectangle(frame, (x - 5, y - text_height - 5), (x + text_width + 5, y + 5), (40, 40, 40), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame

def draw_session_timer(frame: np.ndarray, start_time: float) -> np.ndarray:
    """
    Draws the elapsed session time in the top-left corner.
    
    Args:
        frame: OpenCV image/frame.
        start_time: Session start time.
        
    Returns:
        Modified OpenCV image/frame.
    """
    elapsed = int(time.time() - start_time)
    mins, secs = divmod(elapsed, 60)
    text = f"REC  {mins:02d}:{secs:02d}"
    
    x, y = 10, 30
    cv2.circle(frame, (x + 5, y - 5), 5, (0, 0, 220), -1)
    cv2.putText(frame, text, (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    
    return frame

def draw_alert_border(frame: np.ndarray, triggered: bool) -> np.ndarray:
    """
    Draws a border around the entire frame, turning red when triggered.
    
    Args:
        frame: OpenCV image/frame.
        triggered: Boolean indicating if alert is triggered.
        
    Returns:
        Modified OpenCV image/frame.
    """
    h, w = frame.shape[:2]
    if triggered:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 220), 4)
    else:
        cv2.rectangle(frame, (0, 0), (w, h), (40, 40, 40), 1)
        
    return frame

def draw_class_legend(frame: np.ndarray) -> np.ndarray:
    """
    Draws a static legend in the top-center of the frame showing Living vs Non-living colors.
    
    Args:
        frame: OpenCV image/frame.
        
    Returns:
        Modified OpenCV image/frame.
    """
    h, w = frame.shape[:2]
    text1 = "Living"
    text2 = "Non-living"
    
    (tw1, _), _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    (tw2, _), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
    total_width = 10 + 10 + tw1 + 20 + 10 + tw2 + 10
    x_start = (w - total_width) // 2
    y_start = 10
    height = 25
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_start, y_start), (x_start + total_width, y_start + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    y_text = y_start + 17
    y_circle = y_start + 12
    
    cx1 = x_start + 15
    tx1 = cx1 + 10
    cx2 = tx1 + tw1 + 25
    tx2 = cx2 + 10
    
    cv2.circle(frame, (cx1, y_circle), 5, (0, 0, 220), -1)
    cv2.putText(frame, text1, (tx1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.circle(frame, (cx2, y_circle), 5, (160, 160, 160), -1)
    cv2.putText(frame, text2, (tx2, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def draw_stats(frame: np.ndarray, living_counts: Dict[str, int], nonliving_counts: Dict[str, int], fps: float = 0.0, session_seconds: int = 0, detection_time_ms: float = 0, skip_frames: int = 1) -> np.ndarray:
    """
    Draws the upgraded HUD panel using the provided statistics.
    
    Args:
        frame: OpenCV image/frame.
        living_counts: Dictionary mapping living class names to counts.
        nonliving_counts: Dictionary mapping non-living class names to counts.
        fps: Current frames per second.
        session_seconds: Elapsed session time in seconds.
        detection_time_ms: Latency of the last detection pass.
        skip_frames: Current frame skip setting.
        
    Returns:
        Modified OpenCV image/frame.
    """
    living_count = sum(living_counts.values())
    nonliving_count = sum(nonliving_counts.values())
    return draw_hud_panel(frame, living_count, nonliving_count, fps, session_seconds, detection_time_ms, skip_frames)
