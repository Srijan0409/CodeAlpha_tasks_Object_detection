"""
Custom Centroid-based Object Tracking Module.
Tracks objects across frames by associating their center points.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from collections import OrderedDict, deque, defaultdict
from config import CFG

class CentroidTracker:
    """
    A simple centroid-based tracker that assigns unique IDs to objects.
    """
    def __init__(self, max_disappeared: int = CFG.max_disappeared, distance_threshold: int = CFG.distance_threshold):
        """
        Initializes the tracker.
        
        Args:
            max_disappeared (int): Maximum consecutive frames an object can be lost before deregistration.
            distance_threshold (int): Maximum distance (pixels) to associate a new detection to an existing object.
        """
        self.next_object_id = 1
        self.objects = OrderedDict()       # mapping of id -> centroid
        self.bboxes = OrderedDict()        # mapping of id -> bbox
        self.disappeared = OrderedDict()   # mapping of id -> frames_disappeared
        self.trails: Dict[int, deque] = defaultdict(lambda: deque(maxlen=CFG.trail_length))
        self.newly_registered = set()
        self.max_disappeared = max_disappeared
        self.distance_threshold = distance_threshold
        
    def register(self, centroid: Tuple[int, int], bbox: List[int]):
        """
        Registers a new object.
        """
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.newly_registered.add(self.next_object_id)
        self.next_object_id += 1
        
    def deregister(self, object_id: int):
        """
        Deregisters a lost object.
        """
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.disappeared[object_id]
        if object_id in self.trails:
            del self.trails[object_id]
        
    def _get_centroid(self, bbox: List[int]) -> Tuple[int, int]:
        """
        Calculates the centroid (center) of a bounding box.
        
        Args:
            bbox (List[int]): [x1, y1, x2, y2]
            
        Returns:
            Tuple[int, int]: (center_x, center_y)
        """
        x1, y1, x2, y2 = bbox
        cX = int((x1 + x2) / 2.0)
        cY = int((y1 + y2) / 2.0)
        return (cX, cY)
        
    def get_trail(self, object_id: int) -> list:
        """Return list of (x,y) centroid positions for this object's trail history."""
        return list(self.trails.get(object_id, []))
        
    def update(self, detections: List[Dict[str, Any]]) -> Dict[int, Tuple[int, int, List[int]]]:
        """
        Updates the tracker with new detections.
        
        Args:
            detections (List[Dict[str, Any]]): List of dicts from the detector.
            
        Returns:
            Dict: Dictionary mapping object IDs to (centroid_x, centroid_y, bbox).
        """
        self.newly_registered = set()  # reset each frame
        
        # If no detections, increment disappeared count for all existing objects
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            for object_id, centroid in self.objects.items():
                self.trails[object_id].append(centroid)
            
            return {k: (v[0], v[1], self.bboxes[k]) for k, v in self.objects.items()}
            
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        input_bboxes = []
        
        for i, det in enumerate(detections):
            centroid = self._get_centroid(det["bbox"])
            input_centroids[i] = centroid
            input_bboxes.append(det["bbox"])
            
        # If we are currently not tracking any objects, register all detections
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(tuple(input_centroids[i]), input_bboxes[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # AIML Concept: Euclidean distance for object association
            # Calculate the distance between each existing object and each new detection.
            # We use this distance matrix to find the closest matches.
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            
            # Find the smallest distance in each row, then sort row indices
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                    
                # If distance is greater than threshold, do not associate
                if D[row, col] > self.distance_threshold:
                    continue
                    
                object_id = object_ids[row]
                self.objects[object_id] = tuple(input_centroids[col])
                self.bboxes[object_id] = input_bboxes[col]
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
                
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            
            # Mark unused tracked objects as disappeared
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
                    
            # Register new unassociated detections
            for col in unused_cols:
                self.register(tuple(input_centroids[col]), input_bboxes[col])
                
        for object_id, centroid in self.objects.items():
            self.trails[object_id].append(centroid)
                
        # Return {id: (centroid_x, centroid_y, bbox)}
        return {k: (v[0], v[1], self.bboxes[k]) for k, v in self.objects.items()}
