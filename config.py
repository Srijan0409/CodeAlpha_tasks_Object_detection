import yaml
import os

class Config:
    """
    Configuration class to load and validate settings from a YAML file.
    
    Attributes:
        confidence (float): Minimum confidence threshold for object detection.
        skip_frames (int): Number of frames to skip before running full detection.
        model_path (str): Path to the pretrained YOLOv8 weights file.
        max_disappeared (int): Maximum consecutive frames an object can disappear before it's deregistered.
        distance_threshold (int): Maximum distance between centroids to associate trackings.
        living_color (tuple): BGR color tuple for living objects bounding boxes and text.
        nonliving_color (tuple): BGR color tuple for non-living objects bounding boxes and text.
        label_color (tuple): BGR color tuple for label text.
        show_trails (bool): Whether to show trajectory trails for tracked objects.
        trail_length (int): Maximum number of points in the trajectory trail.
        auto_save_threshold (int): Minimum simultaneous objects to trigger auto-screenshot.
        voice_enabled (bool): Whether to enable voice alerts.
        voice_cooldown (int): Seconds to wait before repeating a voice alert.
    """
    def __init__(self, path="settings.yaml"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"settings.yaml not found at {path}. Please create it.")
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            
        def get_val(section, key):
            try:
                return data[section][key]
            except KeyError as e:
                raise KeyError(f"Missing required config key '{key}' in section '{section}' of settings.yaml") from e
        
        # Detection
        self.confidence = float(get_val('detection', 'confidence_threshold'))
        self.skip_frames = int(get_val('detection', 'skip_frames'))
        self.model_path = str(get_val('detection', 'model_path'))
        
        raw_source = get_val('detection', 'allowed_source')
        try:
            self.allowed_source = int(raw_source)
        except ValueError:
            self.allowed_source = str(raw_source)
        
        # Tracking
        self.max_disappeared = int(get_val('tracking', 'max_disappeared'))
        self.distance_threshold = int(get_val('tracking', 'distance_threshold'))
        
        # Display
        self.living_color = tuple(get_val('display', 'living_color_bgr'))
        self.nonliving_color = tuple(get_val('display', 'nonliving_color_bgr'))
        self.label_color = tuple(get_val('display', 'label_text_color'))
        self.show_trails = bool(get_val('display', 'show_trails'))
        self.trail_length = int(get_val('display', 'trail_length'))
        
        # Recording
        self.auto_save_threshold = int(get_val('recording', 'auto_save_threshold'))
        
        # Alerts
        self.voice_enabled = bool(get_val('alerts', 'voice_enabled'))
        self.voice_cooldown = int(get_val('alerts', 'voice_cooldown_seconds'))
    
    def validate(self):
        assert 0.0 < self.confidence <= 1.0, "confidence_threshold must be between 0 and 1"
        assert self.skip_frames >= 1, "skip_frames must be at least 1"
        assert self.max_disappeared > 0, "max_disappeared must be positive"
        print("Config loaded and validated successfully:")
        for key, value in self.__dict__.items():
            print(f"  {key}: {value}")
        return self

CFG = Config().validate()
