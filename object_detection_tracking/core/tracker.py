"""
Custom Centroid-based Object Tracking Module.

This module provides a simple, fast tracking mechanism that tracks objects
across consecutive frames by associating their center points (centroids).
"""

from collections import OrderedDict, defaultdict, deque
from typing import Any, Dict, List, Tuple
import numpy as np

from core.config import CFG

class CentroidTracker:
    """
    A simple centroid-based tracker that assigns unique IDs to objects.
    
    This tracker maintains active object states and their historical movement paths.
    It links detections between frames based on the minimal spatial displacement of centroids.
    
    Attributes:
        next_object_id (int): The ID counter to assign to the next new object.
        objects (OrderedDict): Mapping of object ID to its current centroid (x, y).
        bboxes (OrderedDict): Mapping of object ID to its current bounding box.
        disappeared (OrderedDict): Mapping of object ID to consecutive frames it has been lost.
        trails (defaultdict): Mapping of object ID to a deque of historical centroids for trails.
        newly_registered (set): A set of object IDs that were registered in the current frame.
        max_disappeared (int): The number of consecutive frames an object can be lost before deregistration.
        distance_threshold (int): The maximum distance between centroids to associate them.
    """
    def __init__(self, max_disappeared: int = CFG.max_disappeared, distance_threshold: int = CFG.distance_threshold):
        """
        Initializes the CentroidTracker.
        
        Args:
            max_disappeared (int): Maximum consecutive frames an object can be lost before deregistration.
            distance_threshold (int): Maximum Euclidean distance (in pixels) to associate a new 
                                      detection to an existing tracked object.
        """
        self.next_object_id = 1
        self.objects = OrderedDict()
        self.bboxes = OrderedDict()
        self.disappeared = OrderedDict()
        self.trails: Dict[int, deque] = defaultdict(lambda: deque(maxlen=CFG.trail_length))
        self.newly_registered = set()
        self.max_disappeared = max_disappeared
        self.distance_threshold = distance_threshold
        
    def register(self, centroid: Tuple[int, int], bbox: List[int]) -> None:
        """
        Registers a newly detected object into the tracking system.
        
        Args:
            centroid (Tuple[int, int]): The (x, y) coordinates of the object's center.
            bbox (List[int]): The bounding box [x1, y1, x2, y2] of the object.
        """
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.newly_registered.add(self.next_object_id)
        self.next_object_id += 1
        
    def deregister(self, object_id: int) -> None:
        """
        Removes an object from tracking after it has been lost for `max_disappeared` frames.
        
        Args:
            object_id (int): The unique ID of the object to deregister.
        """
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.disappeared[object_id]
        if object_id in self.trails:
            del self.trails[object_id]
        
    def _get_centroid(self, bbox: List[int]) -> Tuple[int, int]:
        """
        Calculates the centroid (center point) of a bounding box.
        
        Args:
            bbox (List[int]): Bounding box coordinates formatted as [x1, y1, x2, y2].
            
        Returns:
            Tuple[int, int]: The calculated (center_x, center_y) coordinates.
        """
        x1, y1, x2, y2 = bbox
        cX = int((x1 + x2) / 2.0)
        cY = int((y1 + y2) / 2.0)
        return (cX, cY)
        
    def get_trail(self, object_id: int) -> List[Tuple[int, int]]:
        """
        Retrieves the historical movement path for a given object.
        
        Args:
            object_id (int): The unique ID of the tracked object.
            
        Returns:
            List[Tuple[int, int]]: A list of (x, y) coordinates representing the object's trail.
        """
        return list(self.trails.get(object_id, []))
        
    def update(self, detections: List[Dict[str, Any]]) -> Dict[int, Tuple[int, int, List[int]]]:
        """
        Updates the tracker's state with new detections from the current frame.
        
        AIML Concept: Centroid Tracking Algorithm
        This algorithm works by assuming that an object will move only a small distance 
        between consecutive frames. By calculating the center points (centroids) of all 
        detections in the current frame and comparing them to centroids from the previous 
        frame, we can link bounding boxes to the same object ID.
        
        Args:
            detections (List[Dict[str, Any]]): A list of dictionaries containing detection data
                                               (bounding box, class name, confidence).
            
        Returns:
            Dict[int, Tuple[int, int, List[int]]]: A dictionary mapping active object IDs 
                                                   to their current (centroid_x, centroid_y, bbox).
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
            
            # AIML Concept: Euclidean Distance
            # We compute the pairwise Euclidean distance between all existing tracked 
            # object centroids and all newly detected centroids. This forms a distance 
            # matrix (D) where D[i, j] is the physical distance connecting object i and detection j.
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
                
        return {k: (v[0], v[1], self.bboxes[k]) for k, v in self.objects.items()}
