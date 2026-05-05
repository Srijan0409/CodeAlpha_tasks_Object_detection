import cv2
import time
import os
import datetime
from collections import defaultdict

from detector import Detector
from tracker import CentroidTracker
from utils import draw_detections, draw_fps, draw_stats, calculate_fps

def save_screenshot(frame):
    """
    Saves a screenshot of the current frame.
    
    Args:
        frame: The image frame to save.
        
    Returns:
        str: The filepath of the saved screenshot.
    """
    filepath = datetime.datetime.now().strftime("captures/frame_%H%M%S_%d%b%y.png")
    cv2.imwrite(filepath, frame)
    print(f"Screenshot saved: {filepath}")
    return filepath

def start_recording(frame, fps):
    """
    Starts recording a video clip of the detection loop.
    
    Args:
        frame: The current image frame (used for dimensions).
        fps: The current frames per second (must be at least 10.0).
        
    Returns:
        cv2.VideoWriter: The video writer object, or None if initialization fails.
    """
    filename = datetime.datetime.now().strftime("clips/clip_%H%M%S_%d%b%y.mp4")
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        clip_fps = max(10.0, float(fps))
        writer = cv2.VideoWriter(filename, fourcc, clip_fps, (frame.shape[1], frame.shape[0]))
        print(f"Recording started: {filename}")
        return writer
    except Exception as e:
        print(f"Failed to start recording: {e}")
        return None

def main():
    # 2. SETUP before the main loop starts
    os.makedirs("captures", exist_ok=True)
    os.makedirs("clips", exist_ok=True)

    is_recording = False
    clip_writer = None
    last_auto_save_count = 0
    AUTO_SAVE_THRESHOLD = 10  # auto-save when this many objects detected simultaneously

    # 9. STARTUP MESSAGE
    print("Controls: Q=quit | S=screenshot | R=start/stop recording")
    print(f"Auto-save triggers at {AUTO_SAVE_THRESHOLD}+ simultaneous objects")

    detector = Detector(conf_threshold=0.5)
    tracker = CentroidTracker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    prev_time = time.time()
    current_fps = 0.0
    id_to_class = {}
    id_to_conf = {}
    id_to_living = {}

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect objects
            detections = detector.detect(frame)

            # Update tracker
            tracked_objects = tracker.update(detections)

            # Map metadata for drawing
            bbox_map = {tuple(d["bbox"]): (d["class_name"], d["confidence"], d["is_living"]) for d in detections}
            for obj_id, info in tracked_objects.items():
                cx, cy, bbox = info
                b = tuple(bbox)
                if b in bbox_map:
                    id_to_class[obj_id] = bbox_map[b][0]
                    id_to_conf[obj_id] = bbox_map[b][1]
                    id_to_living[obj_id] = bbox_map[b][2]

            # 7. INSIDE THE MAIN LOOP — auto-save trigger
            current_count = len(tracked_objects)
            if current_count >= AUTO_SAVE_THRESHOLD and current_count != last_auto_save_count:
                save_screenshot(frame)
                last_auto_save_count = current_count
                print(f"Auto-saved: {current_count} objects detected")

            # Calculate stats for the current frame
            living_counts = defaultdict(int)
            nonliving_counts = defaultdict(int)
            
            for obj_id in tracked_objects.keys():
                if obj_id in id_to_class:
                    cls_name = id_to_class[obj_id]
                    if id_to_living.get(obj_id, False):
                        living_counts[cls_name] += 1
                    else:
                        nonliving_counts[cls_name] += 1

            # Draw visualizations
            draw_detections(frame, tracked_objects, id_to_class, id_to_conf, id_to_living)
            draw_stats(frame, living_counts, nonliving_counts)
            
            current_fps, prev_time = calculate_fps(prev_time)
            draw_fps(frame, current_fps)

            # 6. INSIDE THE MAIN LOOP — write to clip if recording
            if is_recording and clip_writer is not None:
                clip_writer.write(frame)
                # Draw a small red dot and "REC" text on frame
                cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Object Detection & Tracking', frame)

            # 5. INSIDE THE MAIN LOOP — keyboard controls
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break  # existing quit
            elif key == ord('s'):
                save_screenshot(frame)
            elif key == ord('r'):
                if not is_recording:
                    clip_writer = start_recording(frame, current_fps)
                    is_recording = True
                    print("Recording started — press R again to stop")
                else:
                    clip_writer.release()
                    clip_writer = None
                    is_recording = False
                    print("Recording stopped and saved")

    finally:
        # 8. CLEANUP in finally block
        if is_recording and clip_writer is not None:
            clip_writer.release()
            print("Recording saved on exit")
            
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
