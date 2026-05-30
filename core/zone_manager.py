from datetime import datetime
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

class ZoneManager:
    """
    Manages polygonal ROI (Region of Interest) zones on the video frame.
    
    Tracks which tracked objects are physically inside each zone boundary,
    handling state changes and raising alerts on initial zone entry.
    Utilizes cv2.pointPolygonTest for precise spatial intersection detection.
    
    Attributes:
        zones (List[Dict[str, Any]]): A list storing the state dictionaries for all active zones.
        drawing (bool): Flag indicating if the user is currently in zone-drawing mode.
        current_points (List[Tuple[int, int]]): Accumulator for points of the polygon being drawn.
        zone_object_history (Dict[int, set]): Mapping to track historical object entries per zone.
    """
    
    def __init__(self) -> None:
        """
        Initializes the ZoneManager with an empty list of zones and state variables.
        """
        self.zones: List[Dict[str, Any]] = []
        self.drawing: bool = False
        self.current_points: List[Tuple[int, int]] = []
        self.zone_object_history: Dict[int, set] = {}
    
    def add_zone(self, points: List[Tuple[int, int]], name: str = None) -> None:
        """
        Registers a newly drawn polygonal zone to the manager.
        
        Args:
            points (List[Tuple[int, int]]): A list of (x, y) coordinates forming the polygon vertices.
            name (str, optional): A descriptive label for this zone. Defaults to "Zone {id}".
        """
        zone_id = len(self.zones) + 1
        self.zones.append({
            'id': zone_id,
            'name': name or f"Zone {zone_id}",
            'points': np.array(points, dtype=np.float32),
            'occupied': False,
            'object_ids_inside': set(),
            'entry_count': 0
        })
        print(f"Zone {zone_id} added with {len(points)} vertices")
    
    def is_point_in_zone(self, zone: Dict[str, Any], cx: int, cy: int) -> bool:
        """
        Evaluates whether a given centroid coordinate lies within a specific zone's boundaries.
        
        Args:
            zone (Dict[str, Any]): The state dictionary of the zone being checked.
            cx (int): The x-coordinate of the object's centroid.
            cy (int): The y-coordinate of the object's centroid.
            
        Returns:
            bool: True if the centroid is strictly inside or exactly on the boundary, False otherwise.
        """
        result = cv2.pointPolygonTest(zone['points'], (float(cx), float(cy)), measureDist=False)
        return result >= 0
    
    def update(self, tracked_objects: Dict[int, Tuple[int, int, List[int]]], id_to_class: Dict[int, str]) -> List[Dict[str, Any]]:
        """
        Checks all currently tracked objects against all active zones to detect intrusions.
        
        Args:
            tracked_objects (Dict[int, Tuple[int, int, List[int]]]): Dictionary mapping object IDs 
                                                                     to (cx, cy, bbox).
            id_to_class (Dict[int, str]): Dictionary mapping object IDs to string class names.
            
        Returns:
            List[Dict[str, Any]]: A list of alert dictionaries for objects that newly entered a zone this frame.
        """
        alerts = []
        
        if not self.zones:
            return alerts
            
        for zone in self.zones:
            currently_inside = set()
            
            for obj_id, obj_data in tracked_objects.items():
                cx, cy, _ = obj_data
                if self.is_point_in_zone(zone, cx, cy):
                    currently_inside.add(obj_id)
                    
                    # New entry \u2014 object was not in zone before
                    if obj_id not in zone['object_ids_inside']:
                        zone['entry_count'] += 1
                        class_name = id_to_class.get(obj_id, 'object')
                        alerts.append({
                            'zone_name': zone['name'],
                            'object_id': obj_id,
                            'class_name': class_name,
                            'event': 'entered'
                        })
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        print(f"[{timestamp}] ALERT: {class_name} #{obj_id} entered {zone['name']}")
            
            zone['object_ids_inside'] = currently_inside
            zone['occupied'] = len(currently_inside) > 0
        
        return alerts
    
    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """
        Renders all active zones onto the provided video frame.
        
        Visual styling:
        - Occupied zones: Red semi-transparent fill with a solid red boundary.
        - Empty zones: Green semi-transparent fill with a solid green boundary.
        
        Args:
            frame (np.ndarray): The OpenCV image/frame to draw on.
            
        Returns:
            np.ndarray: The modified OpenCV image with rendered zones.
        """
        if not self.zones:
            return frame
            
        overlay = frame.copy()
        
        for zone in self.zones:
            pts = zone['points']
            pts_int = pts.astype(np.int32)
            color = (0, 0, 180) if zone['occupied'] else (0, 180, 0)
            fill_color = (0, 0, 80) if zone['occupied'] else (0, 80, 0)
            
            # Fill polygon semi-transparently
            cv2.fillPoly(overlay, [pts_int], fill_color)
            
            # Border
            cv2.polylines(frame, [pts_int], isClosed=True, color=color, thickness=2)
            
            # Label at centroid of zone
            M = cv2.moments(pts_int)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                label = f"{zone['name']} ({len(zone['object_ids_inside'])})"
                cv2.putText(frame, label, (cx-40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        # Blend overlay for transparent fill
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        return frame
    
    def handle_mouse_click(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """
        OpenCV mouse callback handler for drawing polygonal zones interactively.
        
        Args:
            event (int): The OpenCV mouse event type identifier (e.g., cv2.EVENT_LBUTTONDOWN).
            x (int): The x-coordinate of the mouse cursor during the event.
            y (int): The y-coordinate of the mouse cursor during the event.
            flags (int): Any OpenCV mouse event flags.
            param (Any): Additional callback parameters provided by OpenCV.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))
            print(f"Point added: ({x}, {y}) \u2014 right-click to close zone")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_points) >= 3:
                self.add_zone(self.current_points.copy())
                self.current_points = []
            else:
                print("Need at least 3 points to close a zone")
        
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.zones = []
            self.current_points = []
            print("All zones cleared")
    
    def draw_in_progress_zone(self, frame: np.ndarray) -> np.ndarray:
        """
        Renders the active (incomplete) polygon being drawn by the user.
        
        Draws yellow dots at plotted vertices and connects them with lines.
        
        Args:
            frame (np.ndarray): The OpenCV image/frame.
            
        Returns:
            np.ndarray: The modified image with the drawing-in-progress overlay.
        """
        for pt in self.current_points:
            cv2.circle(frame, pt, 5, (0, 255, 255), -1)
        if len(self.current_points) > 1:
            cv2.polylines(frame, [np.array(self.current_points)],
                False, (0, 255, 255), 1)
        return frame
    
    @property
    def zone_count(self) -> int:
        """
        Provides the total number of registered active zones.
        
        Returns:
            int: The integer count of zones.
        """
        return len(self.zones)
    
    def get_summary(self) -> str:
        """
        Generates a multi-line formatted text summary of all zones and their statistics.
        
        Returns:
            str: A formatted string detailing zone names, entry counts, and occupancy status.
        """
        lines = [f"Active zones: {self.zone_count}"]
        for z in self.zones:
            lines.append(f"  {z['name']}: {z['entry_count']} total entries, currently {'OCCUPIED' if z['occupied'] else 'empty'}")
        return "\n".join(lines)
