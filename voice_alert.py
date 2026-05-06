import pyttsx3
import threading
import time

class VoiceAlert:
    """
    Offline text-to-speech alert system using pyttsx3.
    Runs speech synthesis in a background daemon thread.
    Applies per-class cooldown to prevent repetitive announcements.
    """
    
    def __init__(self, enabled: bool = True, cooldown_seconds: int = 3):
        """
        Initializes the VoiceAlert system.
        
        Args:
            enabled (bool): Whether voice alerts are active.
            cooldown_seconds (int): Minimum seconds between announcements for the same class.
        """
        self.enabled = enabled
        self.cooldown = cooldown_seconds
        self.last_spoken: dict = {}  # class_name -> last spoken timestamp
        self._engine = None
        self._lock = threading.Lock()
        
        if self.enabled:
            self._init_engine()
    
    def _init_engine(self):
        """Initialize pyttsx3 engine. Called once at startup."""
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', 160)   # speaking speed (words per minute)
            self._engine.setProperty('volume', 0.9) # volume 0.0 to 1.0
            print("Voice alerts initialized successfully")
        except Exception as e:
            print(f"Voice alert init failed: {e}. Voice alerts disabled.")
            self.enabled = False
    
    def _speak_worker(self, text: str):
        """
        Worker function that runs in a background daemon thread.
        Creates a new pyttsx3 engine instance per thread since it's not thread-safe.
        Never crashes the main loop; silently catches all exceptions.
        
        Args:
            text (str): The text to be spoken.
        """
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            pass  # silent fail \u2014 never crash detection loop
    
    def announce(self, class_name: str, is_new_id: bool = False):
        """
        Speak the class name if cooldown has passed.
        Runs the actual TTS synthesis in a non-blocking daemon thread.
        
        Args:
            class_name (str): YOLO class like 'person', 'car', 'dog'
            is_new_id (bool): True if this is a brand new tracking ID (first detection)
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
    
    def announce_zone_alert(self, class_name: str, zone_name: str):
        """
        Speak a zone intrusion alert.
        Runs the actual TTS synthesis in a non-blocking daemon thread.
        
        Args:
            class_name (str): The detected class name.
            zone_name (str): The name of the zone entered.
        """
        if not self.enabled:
            return
        text = f"Alert: {class_name} entered {zone_name}"
        t = threading.Thread(target=self._speak_worker, args=(text,), daemon=True)
        t.start()
    
    def set_enabled(self, state: bool):
        """
        Toggle voice on/off at runtime.
        
        Args:
            state (bool): True to enable, False to disable.
        """
        self.enabled = state
        status = "enabled" if state else "disabled"
        print(f"Voice alerts {status}")
    
    def toggle(self):
        """Toggle voice on/off."""
        self.set_enabled(not self.enabled)
