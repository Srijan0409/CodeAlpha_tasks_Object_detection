import csv
import datetime
import os
import time
from typing import Dict, Any

class DetectionLogger:
    """
    Logs every tracked detection to a CSV file.
    
    Each row captures: timestamp, frame number, tracking ID, class name,
    confidence score, centroid coordinates, bounding box, and living/non-living label.
    
    The CSV is written incrementally (one row per detection per frame) so data
    is not lost if the program crashes before session ends.
    
    Attributes:
        COLUMNS (List[str]): The column headers for the CSV file.
        filepath (str): The absolute or relative path to the output CSV file.
        row_count (int): The total number of rows logged in the current session.
        session_start (float): The timestamp when the logger was initialized.
    """
    
    COLUMNS = [
        'timestamp', 'frame_num', 'tracking_id', 'class_name',
        'is_living', 'confidence', 'centroid_x', 'centroid_y',
        'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'
    ]
    
    def __init__(self, output_dir: str = "logs"):
        """
        Initializes the DetectionLogger and creates the CSV file with headers.
        
        Args:
            output_dir (str): The directory where the log file will be saved.
        """
        os.makedirs(output_dir, exist_ok=True)
        session_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(output_dir, f"session_{session_ts}.csv")
        
        self._file = open(self.filepath, 'w', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=self.COLUMNS)
        self._writer.writeheader()
        self._file.flush()
        
        self.row_count = 0
        self.session_start = time.time()
        print(f"Detection log started: {self.filepath}")
    
    def log_frame(self, frame_num: int, tracked_objects: Dict[int, Any], id_to_class: Dict[int, str], id_to_conf: Dict[int, float], id_to_living: Dict[int, bool]) -> None:
        """
        Logs all tracked objects from a single frame to the CSV file.
        
        Args:
            frame_num (int): The sequential number of the current frame.
            tracked_objects (Dict[int, Any]): Dictionary mapping tracking IDs to (centroid_x, centroid_y, bounding_box).
            id_to_class (Dict[int, str]): Dictionary mapping tracking IDs to class labels.
            id_to_conf (Dict[int, float]): Dictionary mapping tracking IDs to confidence scores.
            id_to_living (Dict[int, bool]): Dictionary mapping tracking IDs to a living/non-living boolean flag.
        """
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        for obj_id, info in tracked_objects.items():
            cx, cy, bbox = info
            
            row = {
                'timestamp':   ts,
                'frame_num':   frame_num,
                'tracking_id': obj_id,
                'class_name':  id_to_class.get(obj_id, 'unknown'),
                'is_living':   id_to_living.get(obj_id, False),
                'confidence':  round(id_to_conf.get(obj_id, 0.0), 3),
                'centroid_x':  cx,
                'centroid_y':  cy,
                'bbox_x1':     bbox[0],
                'bbox_y1':     bbox[1],
                'bbox_x2':     bbox[2],
                'bbox_y2':     bbox[3],
            }
            self._writer.writerow(row)
            self.row_count += 1
        
        # Flush every 100 rows to prevent data loss on crash
        if self.row_count % 100 == 0:
            self._file.flush()
    
    def close(self) -> None:
        """
        Closes the CSV file safely and prints a summary of the session to the console.
        """
        self._file.flush()
        self._file.close()
        duration = round(time.time() - self.session_start, 1)
        print(f"Detection log saved: {self.filepath}")
        print(f"Total rows logged: {self.row_count} over {duration}s")
        print(f"Open in Excel or pandas: pd.read_csv('{self.filepath}')")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieves the current logging statistics without closing the logger.
        
        Returns:
            Dict[str, Any]: A dictionary containing 'rows_logged', 'filepath', and 'duration_s'.
        """
        return {
            'rows_logged': self.row_count,
            'filepath': self.filepath,
            'duration_s': round(time.time() - self.session_start, 1)
        }
