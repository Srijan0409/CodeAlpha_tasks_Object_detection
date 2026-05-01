"""
Main entry point for Real-Time Object Detection and Tracking System.
"""

import argparse
import cv2
import sys
import time
from collections import defaultdict
from detector import Detector
from tracker import CentroidTracker
import utils

def main():
    parser = argparse.ArgumentParser(description="Real-Time Object Detection and Tracking")
    parser.add_argument("--source", type=str, default="0", help="Source for video (0 for webcam, or path to video file)")
    parser.add_argument("--save", action="store_true", help="Flag to save output as output.mp4")
    args = parser.parse_args()

    print("🚀 Initializing system...")

    # Determine source (int for webcam, str for file)
    source = int(args.source) if args.source.isdigit() else args.source

    # Handle VideoCapture failure gracefully
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source '{args.source}'. Please check the path or your webcam connection.")
        sys.exit(1)

    print(f"✅ Successfully connected to source: {args.source}")

    # Initialize Detector and Tracker
    detector = Detector()
    tracker = CentroidTracker()

    # Setup video writer if save flag is true
    out = None
    if args.save:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30 # Default if cannot be read
        
        out_path = "output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
        print(f"💾 Saving output to {out_path}")

    prev_time = time.time()
    print("🎯 Detection started... Press 'q' to quit.")
    
    id_to_class = {}
    id_to_conf = {}

    # AIML Concept: Real-time frame processing pipeline
    # We continually fetch frames, run the model inference, update the tracker, and render the output.
    while True:
        ret, frame = cap.read()
        if not ret:
            print("🎬 End of video stream.")
            break

        # 1. Detect objects
        detections = detector.detect(frame)
        
        # Helper to map bboxes back to classes and confidence
        bbox_map = {tuple(d["bbox"]): (d["class_name"], d["confidence"]) for d in detections}

        # 2. Update tracker
        tracked_objects = tracker.update(detections)

        # Update metadata maps for drawing
        for obj_id, info in tracked_objects.items():
            _, _, bbox = info
            b = tuple(bbox)
            if b in bbox_map:
                id_to_class[obj_id] = bbox_map[b][0]
                id_to_conf[obj_id] = bbox_map[b][1]

        # 3. Calculate statistics
        object_count_dict = defaultdict(int)
        for obj_id in tracked_objects.keys():
            if obj_id in id_to_class:
                object_count_dict[id_to_class[obj_id]] += 1

        # 4. Draw visualizations
        utils.draw_detections(frame, tracked_objects, id_to_class, id_to_conf)
        
        fps, prev_time = utils.calculate_fps(prev_time)
        utils.draw_fps(frame, fps)
        
        utils.draw_stats(frame, object_count_dict)

        # Save frame if requested
        if args.save and out is not None:
            out.write(frame)

        # Show frame
        cv2.imshow("Real-Time Tracking", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Interrupted by user.")
            break

    # Proper cleanup
    print("🧹 Cleaning up resources...")
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print("👋 Goodbye!")

if __name__ == "__main__":
    main()
