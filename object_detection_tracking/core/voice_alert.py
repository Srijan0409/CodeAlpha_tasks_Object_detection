import threading
import time
import pyttsx3

class VoiceAlert:
    """
    Offline text-to-speech alert system using pyttsx3.
    
    Runs speech synthesis in a background daemon thread so it doesn't block the
    main video processing loop. Applies a per-class cooldown to prevent repetitive,
    spammy announcements.
    
    Attributes:
        enabled (bool): Status flag indicating if voice alerts are currently active.
        cooldown (int): The minimum time (in seconds) between consecutive alerts for the same class.
        last_spoken (dict): Dictionary tracking the last spoken timestamp for each class.
    """
    
    def __init__(self, enabled: bool = True, cooldown_seconds: int = 3):
        """
        Initializes the VoiceAlert system.
        
        Args:
            enabled (bool): Whether voice alerts are active on startup.
            cooldown_seconds (int): Minimum seconds between announcements for the same class.
        """
        self.enabled = enabled
        self.cooldown = cooldown_seconds
        self.last_spoken: dict = {}  # class_name -> last spoken timestamp
        self._engine = None
        self._lock = threading.Lock()
        
        if self.enabled:
            self._init_engine()
    
    def _init_engine(self) -> None:
        """
        Initializes the pyttsx3 engine. Called once during class instantiation.
        If initialization fails, it automatically disables voice alerts.
        """
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', 160)   # speaking speed (words per minute)
            self._engine.setProperty('volume', 0.9) # volume 0.0 to 1.0
            print("Voice alerts initialized successfully")
        except Exception as e:
            print(f"Voice alert init failed: {e}. Voice alerts disabled.")
            self.enabled = False
    
    def _speak_worker(self, text: str) -> None:
        """
        Worker function that executes speech synthesis in a background daemon thread.
        
        Creates a new pyttsx3 engine instance per thread since the library is not thread-safe.
        Silently catches exceptions to ensure the main tracking loop never crashes.
        
        Args:
            text (str): The text phrase to be spoken aloud.
        """
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            pass  # silent fail \u2014 never crash detection loop
    
    def announce(self, class_name: str, is_new_id: bool = False) -> None:
        """
        Speaks the class name if the designated cooldown period has passed.
        
        Args:
            class_name (str): The YOLO class label (e.g., 'person', 'car', 'dog').
            is_new_id (bool): True if this is a newly registered tracking ID.
        """
        if not self.enabled:
            return
        
        with self._lock:
            now = time.time()
            last = self.last_spoken.get(class_name, 0)
            
            if now - last < self.cooldown:
                return  # still in cooldown period
            
            self.last_spoken[class_name] = now
            
        # Build speech text
        if is_new_id:
            text = f"New {class_name} detected"
        else:
            text = f"{class_name}"
        
        # Run in daemon thread \u2014 dies automatically when main program exits
        t = threading.Thread(target=self._speak_worker, args=(text,), daemon=True)
        t.start()
    
    def announce_zone_alert(self, class_name: str, zone_name: str) -> None:
        """
        Speaks a targeted alert when an object enters a defined Region of Interest (ROI) zone.
        
        Args:
            class_name (str): The detected class name that triggered the alert.
            zone_name (str): The name of the zone that was intruded.
        """
        if not self.enabled:
            return
        text = f"Alert: {class_name} entered {zone_name}"
        t = threading.Thread(target=self._speak_worker, args=(text,), daemon=True)
        t.start()
    
    def set_enabled(self, state: bool) -> None:
        """
        Sets the active state of the voice alerts.
        
        Args:
            state (bool): True to enable voice alerts, False to mute them.
        """
        self.enabled = state
        status = "enabled" if state else "disabled"
        print(f"Voice alerts {status}")
    
    def toggle(self) -> None:
        """
        Toggles the current state of voice alerts between enabled and disabled.
        """
        self.set_enabled(not self.enabled)
