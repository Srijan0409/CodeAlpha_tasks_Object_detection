"""
YOLOv8-based Object Detection Module.

This module handles loading the pretrained YOLOv8 model and running inference
on image or video frames to detect objects.
"""

from typing import Any, Dict, List
import numpy as np
from ultralytics import YOLO

from core.config import CFG

# New - detect everything, classify as living or non-living
LIVING_CLASSES = {
    'person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe'
}

class Detector:
    """
    Wrapper class for YOLOv8 object detection.
    
    This class loads a pre-trained YOLO model and processes frames to return
    bounding boxes, confidences, class names, and a custom 'is_living' flag.
    
    Attributes:
        model (YOLO): The loaded YOLOv8 model instance.
        conf_threshold (float): Minimum confidence score to keep a detection.
    """
    def __init__(self, model_path: str = CFG.model_path, conf_threshold: float = CFG.confidence):
        """
        Initializes the YOLOv8 detector.
        
        Args:
            model_path (str): Path to the pretrained YOLOv8 weights (e.g., 'yolov8n.pt').
            conf_threshold (float): Minimum confidence threshold for filtering detections.
        """
        print(f"🎯 Loading YOLO model from {model_path}...")
        
        # AIML Concept: Convolutional Neural Networks (CNN) backbone
        # The YOLO architecture utilizes a deep CNN backbone (like CSPDarknet) 
        # to extract rich, hierarchical spatial features from the input frames.
        
        # AIML Concept: Anchor-free object detection
        # Modern iterations like YOLOv8 are anchor-free models. They directly predict 
        # object centers and bounding box dimensions, avoiding complex anchor box tuning.
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs object detection on a single frame.
        
        Args:
            frame (np.ndarray): The image or video frame from OpenCV (BGR format).
            
        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing valid detections.
                Each dictionary contains:
                - 'bbox' (List[int]): Bounding box coordinates [x1, y1, x2, y2].
                - 'class_name' (str): The string label of the detected object class.
                - 'confidence' (float): The detection confidence score.
                - 'is_living' (bool): True if the class is designated as a living entity.
        """
        # AIML Concept: Non-Maximum Suppression (NMS)
        # The detect function inherently relies on NMS applied by the Ultralytics backend.
        # NMS filters out multiple overlapping bounding boxes that predict the same object,
        # keeping only the box with the highest confidence score.
        results = self.model(frame, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Filter by confidence
            if conf >= self.conf_threshold:
                # AIML Concept: Bounding box regression
                # Instead of standard classification, YOLO performs bounding box regression.
                # It outputs continuous coordinate values to pinpoint the bounding box limits.
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
