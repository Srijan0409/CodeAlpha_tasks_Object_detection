import datetime
import os
import time
from collections import defaultdict

import cv2

from core.camera_stream import CameraStream
from core.config import CFG
from core.detection_logger import DetectionLogger
from core.detector import Detector
from core.heatmap import DetectionHeatmap
from core.tracker import CentroidTracker
from core.utils import (calculate_fps, draw_detections, draw_fps, draw_stats,
                        draw_trails)
from core.voice_alert import VoiceAlert
from core.zone_manager import ZoneManager


def save_screenshot(frame):
    """
    Saves a screenshot of the current frame.
    
    Args:
        frame: The image frame to save.
        
    Returns:
        str: The filepath of the saved screenshot.
    """
    filepath = datetime.datetime.now().strftime("output/captures/frame_%H%M%S_%d%b%y.png")
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
    filename = datetime.datetime.now().strftime("output/clips/clip_%H%M%S_%d%b%y.mp4")
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
    os.makedirs("output/captures", exist_ok=True)
    os.makedirs("output/clips", exist_ok=True)

    is_recording = False
    clip_writer = None
    last_auto_save_count = 0
    AUTO_SAVE_THRESHOLD = CFG.auto_save_threshold  # auto-save when this many objects detected simultaneously

    # 9. STARTUP MESSAGE
    print("Controls: Q=quit | S=screenshot | R=start/stop recording")
    print("Z = zone mode | Draw zones by clicking on the video window")
    print(f"Auto-save triggers at {AUTO_SAVE_THRESHOLD}+ simultaneous objects")
    skip_frames = getattr(CFG, 'skip_frames', 2)
    print(f"Frame skip: {skip_frames} \u2014 YOLO runs every {skip_frames} frames")
    print("Press +/- to adjust frame skip live")
    print("H = toggle heatmap | C = clear heatmap")

    detector = Detector(conf_threshold=CFG.confidence)
    tracker = CentroidTracker()
    zone_mgr = ZoneManager()
    voice = VoiceAlert(enabled=CFG.voice_enabled, cooldown_seconds=CFG.voice_cooldown)
    print("Voice alerts: ON \u2014 press V to toggle" if CFG.voice_enabled else "Voice alerts: OFF \u2014 press V to toggle")
    
    logger = DetectionLogger(output_dir="output/logs")
    print(f"Logging detections to: {logger.filepath}")
    print("Press L to see log stats")

    cv2.namedWindow('Object Detection & Tracking', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Object Detection & Tracking', zone_mgr.handle_mouse_click)

    try:
        stream = CameraStream(source=CFG.allowed_source).start()
        w, h = stream.get_resolution()
        heatmap = DetectionHeatmap(frame_height=h, frame_width=w, decay=0.998)
        print(f"Resolution: {w}x{h}")
        print(f"Threaded capture active \u2014 source FPS: {stream.get_source_fps():.0f}")
    except RuntimeError as e:
        print(f"Camera error: {e}")
        return

    prev_time = time.time()
    session_start_time = time.time()
    current_fps = 0.0
    id_to_class = {}
    id_to_conf = {}
    id_to_living = {}
    frame_count = 0
    show_trails = CFG.show_trails
    last_detections = []
    detection_time_ms = 0

    try:
        while stream.is_running():
            ret, frame = stream.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            frame_count += 1
            
            if frame_count % skip_frames == 0:
                t_start = time.time()
                last_detections = detector.detect(frame)
                detection_time_ms = (time.time() - t_start) * 1000
            
            # Always update tracker \u2014 even on skipped frames
            # Tracker uses Euclidean distance to keep IDs stable between YOLO frames
            tracked_objects = tracker.update(last_detections)

            # Map metadata for drawing
            bbox_map = {tuple(d["bbox"]): (d["class_name"], d["confidence"], d["is_living"]) for d in last_detections}
            for obj_id, info in tracked_objects.items():
                cx, cy, bbox = info
                b = tuple(bbox)
                if b in bbox_map:
                    id_to_class[obj_id] = bbox_map[b][0]
                    id_to_conf[obj_id] = bbox_map[b][1]
                    id_to_living[obj_id] = bbox_map[b][2]

            for obj_id, obj_data in tracked_objects.items():
                cx, cy, _ = obj_data
                conf = id_to_conf.get(obj_id, 1.0)
                heatmap.add_detection(cx, cy, confidence=conf)
                
            heatmap.decay_frame()

            logger.log_frame(frame_count, tracked_objects, id_to_class, id_to_conf, id_to_living)

            newly_registered_ids = tracker.newly_registered
            for obj_id, obj_data in tracked_objects.items():
                if obj_id in newly_registered_ids:
                    cls_name = id_to_class.get(obj_id, "object")
                    voice.announce(cls_name, is_new_id=True)

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
            if heatmap.enabled:
                frame = heatmap.render(frame, alpha=0.45)
            if show_trails:
                frame = draw_trails(frame, tracker, tracked_objects, id_to_living)
            draw_detections(frame, tracked_objects, id_to_class, id_to_conf, id_to_living)
            
            frame = zone_mgr.draw_zones(frame)
            frame = zone_mgr.draw_in_progress_zone(frame)
            alerts = zone_mgr.update(tracked_objects, id_to_class)
            for alert in alerts:
                if alert['event'] == 'entered':
                    voice.announce_zone_alert(alert['class_name'], alert['zone_name'])
            
            session_seconds = int(time.time() - session_start_time)
            draw_stats(frame, living_counts, nonliving_counts, current_fps, session_seconds, detection_time_ms, skip_frames)
            
            current_fps, prev_time = calculate_fps(prev_time)
            draw_fps(frame, current_fps)
            
            if frame_count % 300 == 0:
                effective_det_fps = current_fps / skip_frames
                print(f"Frame {frame_count} | Display FPS: {current_fps:.1f} | Detection FPS: {effective_det_fps:.1f} | Det latency: {detection_time_ms:.0f}ms")

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
            elif key == ord('z'):
                print("Zone mode: left-click to add points, right-click to close, middle-click to clear")
            elif key == ord('+') or key == ord('='):
                skip_frames = min(skip_frames + 1, 5)
                print(f"Frame skip: {skip_frames} (YOLO runs every {skip_frames} frames)")
            elif key == ord('-'):
                skip_frames = max(skip_frames - 1, 1)
                print(f"Frame skip: {skip_frames} (YOLO runs every {skip_frames} frames)")
            elif key == ord('t'):
                show_trails = not show_trails
                print(f"Trails: {'ON' if show_trails else 'OFF'}")
            elif key == ord('l'):
                stats = logger.get_stats()
                print(f"Log stats: {stats['rows_logged']} rows | File: {stats['filepath']}")
            elif key == ord('h'):
                heatmap.toggle()
            elif key == ord('c'):
                heatmap.reset()
                print("Heatmap cleared")
            elif key == ord('v'):
                voice.toggle()
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
        print(zone_mgr.get_summary())
        logger.close()
        
        # 8. CLEANUP in finally block
        if is_recording and clip_writer is not None:
            clip_writer.release()
            print("Recording saved on exit")
            
        stream.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
