import cv2
import threading
import time
from typing import Optional, Union

class CameraStream:
    """
    Threaded video capture wrapper for OpenCV.
    
    Reads frames in a background daemon thread continuously.
    Main thread always gets the latest frame immediately without blocking.
    This eliminates the frame-read bottleneck that causes lag on CPU-only systems.
    
    Supports both webcam index (int) and video file path (str).
    """
    
    def __init__(self, source: Union[int, str] = 0):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}. Check webcam connection or file path.")
        
        self.frame: Optional[any] = None
        self.running = False
        self._lock = threading.Lock()
        self._thread = None
        
        # Read one frame to initialize
        ret, self.frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read first frame from source.")
        
        print(f"CameraStream initialized: source={source}, resolution={self.get_resolution()}")
    
    def start(self) -> 'CameraStream':
        """Start the background frame reading thread. Returns self for chaining."""
        self.running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        print("CameraStream background thread started")
        return self
    
    def _reader_loop(self):
        """
        Background thread loop.
        Continuously reads frames and stores the latest one.
        Runs until self.running is False.
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
    
    def read(self):
        """
        Return the latest frame captured by the background thread.
        Never blocks \u2014 always returns immediately.
        Returns (True, frame) or (False, None) to match cv2.VideoCapture.read() interface.
        """
        with self._lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()
    
    def get_resolution(self) -> tuple:
        """Return (width, height) of the video source."""
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)
    
    def get_source_fps(self) -> float:
        """Return the FPS reported by the video source (0 for webcams)."""
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    def is_running(self) -> bool:
        """Return True if the background thread is still reading frames."""
        return self.running
    
    def stop(self):
        """Stop the background thread and release the video capture."""
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.cap.release()
        print("CameraStream stopped and released")
    
    def __enter__(self):
        return self.start()
    
    def __exit__(self, *args):
        self.stop()
