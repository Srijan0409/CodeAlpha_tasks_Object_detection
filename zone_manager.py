import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from datetime import datetime

class ZoneManager:
    """
    Manages polygonal ROI zones on the video frame.
    Tracks which objects are inside each zone and raises alerts on entry.
    Uses cv2.pointPolygonTest for precise polygon intersection detection.
    """
    
    def __init__(self) -> None:
        """Initialize the ZoneManager with an empty list of zones and state variables."""
        self.zones: List[Dict[str, Any]] = []
        self.drawing: bool = False
        self.current_points: List[Tuple[int, int]] = []
        self.zone_object_history: Dict[int, set] = {}
    
    def add_zone(self, points: List[Tuple[int, int]], name: str = None) -> None:
        """
        Add a new polygonal zone.
        
        Args:
            points: List of (x,y) tuples forming the polygon vertices.
            name: Optional label for this zone e.g. "Zone A".
        """
        zone_id = len(self.zones) + 1
        self.zones.append({
            'id': zone_id,
            'name': name or f"Zone {zone_id}",
            'points': np.array(points, dtype=np.int32),
            'occupied': False,
            'object_ids_inside': set(),
            'entry_count': 0
        })
        print(f"Zone {zone_id} added with {len(points)} vertices")
    
    def is_point_in_zone(self, zone: Dict[str, Any], cx: int, cy: int) -> bool:
        """
        Check if centroid (cx, cy) is inside the zone polygon.
        
        Args:
            zone: The zone dictionary containing polygon points.
            cx: Centroid x-coordinate.
            cy: Centroid y-coordinate.
            
        Returns:
            bool: True if inside or on boundary.
        """
        result = cv2.pointPolygonTest(zone['points'], (float(cx), float(cy)), measureDist=False)
        return result >= 0
    
    def update(self, tracked_objects: Dict[int, Tuple[int, int, List[int]]], id_to_class: Dict[int, str]) -> List[Dict[str, Any]]:
        """
        Check all tracked objects against all zones.
        
        Args:
            tracked_objects: Dictionary mapping object ID to (cx, cy, bbox).
            id_to_class: Dictionary mapping object ID to class name.
            
        Returns:
            List[Dict]: List of alert dictionaries for newly entered objects.
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
        Draw all zones on the frame.
        Occupied zones: red fill + red border.
        Empty zones: green semi-transparent fill + green border.
        Each zone shows its name and object count inside.
        
        Args:
            frame: OpenCV image/frame.
            
        Returns:
            np.ndarray: Modified OpenCV image/frame.
        """
        if not self.zones:
            return frame
            
        overlay = frame.copy()
        
        for zone in self.zones:
            pts = zone['points']
            color = (0, 0, 180) if zone['occupied'] else (0, 180, 0)
            fill_color = (0, 0, 80) if zone['occupied'] else (0, 80, 0)
            
            # Fill polygon semi-transparently
            cv2.fillPoly(overlay, [pts], fill_color)
            
            # Border
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            
            # Label at centroid of zone
            M = cv2.moments(pts)
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
        Mouse callback for drawing zones interactively.
        
        Args:
            event: OpenCV mouse event type.
            x: x-coordinate of the mouse click.
            y: y-coordinate of the mouse click.
            flags: OpenCV mouse event flags.
            param: Additional parameters.
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
        Draw dots and lines showing the zone currently being drawn.
        
        Args:
            frame: OpenCV image/frame.
            
        Returns:
            np.ndarray: Modified OpenCV image/frame.
        """
        for pt in self.current_points:
            cv2.circle(frame, pt, 5, (0, 255, 255), -1)
        if len(self.current_points) > 1:
            cv2.polylines(frame, [np.array(self.current_points)],
                False, (0, 255, 255), 1)
        return frame
    
    @property
    def zone_count(self) -> int:
        """Returns the number of active zones."""
        return len(self.zones)
    
    def get_summary(self) -> str:
        """
        Generate a text summary of all zones and their statistics.
        
        Returns:
            str: Multi-line summary string.
        """
        lines = [f"Active zones: {self.zone_count}"]
        for z in self.zones:
            lines.append(f"  {z['name']}: {z['entry_count']} total entries, currently {'OCCUPIED' if z['occupied'] else 'empty'}")
        return "\n".join(lines)
