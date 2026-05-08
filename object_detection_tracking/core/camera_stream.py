import threading
import time
from typing import Optional, Union, Tuple
import cv2
import numpy as np

class CameraStream:
    """
    Threaded video capture wrapper for OpenCV.
    
    Reads frames in a background daemon thread continuously.
    Main thread always gets the latest frame immediately without blocking.
    This eliminates the frame-read bottleneck that causes lag on CPU-only systems.
    
    Attributes:
        source (Union[int, str]): The video source (e.g., 0 for webcam, filepath for video).
        cap (cv2.VideoCapture): The OpenCV video capture object.
        frame (Optional[np.ndarray]): The most recently read video frame.
        running (bool): Flag indicating whether the background thread is actively running.
    """
    
    def __init__(self, source: Union[int, str] = 0):
        """
        Initializes the CameraStream.
        
        Args:
            source (Union[int, str]): The device index (int) or file path (str) for the video source.
            
        Raises:
            RuntimeError: If the video source cannot be opened or the first frame fails to read.
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}. Check webcam connection or file path.")
        
        self.frame: Optional[np.ndarray] = None
        self.running = False
        self._lock = threading.Lock()
        self._thread = None
        
        # Read one frame to initialize
        ret, self.frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read first frame from source.")
        
        print(f"CameraStream initialized: source={source}, resolution={self.get_resolution()}")
    
    def start(self) -> 'CameraStream':
        """
        Starts the background frame reading thread.
        
        Returns:
            CameraStream: Returns self for method chaining.
        """
        self.running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        print("CameraStream background thread started")
        return self
    
    def _reader_loop(self) -> None:
        """
        Background thread loop for continuous frame reading.
        
        Continuously reads frames from the video capture object and stores the latest 
        one in `self.frame`. Runs until `self.running` is set to False.
        """
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    self.running = False
                    break
                with self._lock:
                    self.frame = frame
                time.sleep(0.001)  # tiny sleep to prevent CPU spinning at 100%
        except Exception as e:
            print(f"CameraStream thread error: {e}")
            self.running = False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Returns the latest frame captured by the background thread.
        
        This method never blocks; it always returns immediately with the most recent frame.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: A tuple where the first element is a boolean 
                                               indicating success, and the second is the frame array.
        """
        with self._lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()
    
    def get_resolution(self) -> Tuple[int, int]:
        """
        Retrieves the resolution of the video source.
        
        Returns:
            Tuple[int, int]: The (width, height) of the video frames.
        """
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)
    
    def get_source_fps(self) -> float:
        """
        Retrieves the frames per second (FPS) reported by the video source.
        
        Returns:
            float: The FPS of the source video (often 0.0 for live webcams).
        """
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    def is_running(self) -> bool:
        """
        Checks if the background reading thread is currently active.
        
        Returns:
            bool: True if running, False otherwise.
        """
        return self.running
    
    def stop(self) -> None:
        """
        Stops the background thread and safely releases the video capture object.
        """
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.cap.release()
        print("CameraStream stopped and released")
    
    def __enter__(self) -> 'CameraStream':
        """Context manager entry point."""
        return self.start()
    
    def __exit__(self, *args) -> None:
        """Context manager exit point."""
        self.stop()
