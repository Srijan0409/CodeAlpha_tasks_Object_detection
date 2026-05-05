"""
YOLOv8-based Object Detection Module.
This module handles loading the pretrained YOLOv8 model and running inference.
"""

from ultralytics import YOLO
from typing import List, Dict, Any
import numpy as np
from config import CFG

# New - detect everything, classify as living or non-living
LIVING_CLASSES = [
    'person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe'
]

class Detector:
    """
    Wrapper class for YOLOv8 object detection.
    Filters detections by confidence and specific classes.
    """
    def __init__(self, model_path: str = CFG.model_path, conf_threshold: float = CFG.confidence):
        """
        Initializes the YOLOv8 detector.
        
        Args:
            model_path (str): Path to the pretrained YOLOv8 weights.
            conf_threshold (float): Minimum confidence to keep a detection.
        """
        print(f"🎯 Loading YOLO model from {model_path}...")
        # AIML Concept: Convolutional Neural Networks (CNN) backbone
        # YOLOv8 utilizes a CNN backbone to extract rich feature maps from images.
        # AIML Concept: Anchor-free object detection
        # YOLOv8 is an anchor-free model, directly predicting object centers and bounding box dimensions.
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        pass
        
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs object detection on a single frame.
        
        Args:
            frame (np.ndarray): The image/video frame from OpenCV.
            
        Returns:
            List[Dict[str, Any]]: List of detections. Format:
                {bbox: [x1, y1, x2, y2], class_name: str, confidence: float}
        """
        # Run inference
        # AIML Concept: Non-Maximum Suppression (NMS)
        # Ultralytics internally applies NMS to filter out overlapping bounding boxes predicting the same object.
        results = self.model(frame, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Filter by confidence
            if conf >= self.conf_threshold:
                # AIML Concept: Bounding box regression
                # YOLO regresses the final bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_name = self.model.names[cls_id]
                is_living = class_name in LIVING_CLASSES
                
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "class_name": class_name,
                    "confidence": conf,
                    "is_living": is_living
                })
                
        return detections
