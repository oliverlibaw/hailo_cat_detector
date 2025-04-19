# This project detects cats and triggers a water gun to squirt them. It runs on a Raspberry Pi 5 with a Hailo AI Kit and a four-relay HAT.

import os
import cv2
import time
import numpy as np
import random
import torch
from ultralytics import YOLO
import pygame
import signal
import sys
import datetime

# Development mode flag - set to True when developing on MacBook
DEV_MODE = False  # Set to False for Raspberry Pi deployment

# Print the version of OpenCV being used
print(f"OpenCV version: {cv2.__version__}")

# Check if running in headless mode (without display)
HEADLESS_MODE = True  # Always run in headless mode for now
print("Running in headless mode - terminal output only")

# Import Raspberry Pi specific modules only in production mode
if not DEV_MODE:
    try:
        from picamera2 import Picamera2
        import RPi.GPIO as GPIO
        print("Successfully imported Pi-specific modules")
    except ImportError as e:
        print(f"Error importing Pi-specific modules: {e}")
        print("Please ensure you have the following packages installed:")
        print("sudo apt install -y python3-libcamera python3-picamera2")
        print("And that you're running the script with the correct Python environment")
        raise

# GPIO Pin Setup
RELAY_PINS = {
    'squirt': 16,    # Squirt relay (triggers water gun)
    'left': 6,      # Left relay (triggers for left-side movement)
    'right': 13,    # Right relay (triggers for right-side movement)
    'unused': 15    # Unused relay
}

# Important: Set to True if your relay module activates on LOW rather than HIGH
RELAY_ACTIVE_LOW = True    # Many relay HATs activate on LOW signal

# This flag indicates relays are "normally closed" - they're ON when not activated
RELAY_NORMALLY_CLOSED = False  # Set to False since we're hearing the relay click when activated

# IMPORTANT: The squirt relay has opposite behavior from the other relays
# For squirt relay: HIGH = ON, LOW = OFF
# For other relays: LOW = ON, HIGH = OFF (standard active-low relay behavior)
SQUIRT_RELAY_ON_STATE = GPIO.HIGH   # Set to HIGH to turn the squirt relay ON
SQUIRT_RELAY_OFF_STATE = GPIO.LOW    # Set to LOW to turn the squirt relay OFF

# Model Setup
inference_host_address = "@local"
zoo_url = "/home/pi5/degirum_model_zoo"
token = ""
model_name = "yolo11s_silu_coco--640x640_quant_hailort_hailo8l_1"  # YOLO11s model - larger and more accurate

# Configuration
DETECTION_THRESHOLD = 0.30  # Confidence threshold for detections (lowered from 0.40 for better recall)
MODEL_INPUT_SIZE = (640, 640)  # YOLOv11 input size
CENTER_THRESHOLD = 0.1  # Threshold for determining if object is left/right of center
RELAY_SQUIRT_DURATION = 0.2  # Duration to activate squirt relay in seconds
RELAY_SQUIRT_COOLDOWN = 1.0  # Cooldown period for squirt relay in seconds
RELAY_HYSTERESIS_TIME = 0.5  # Minimum time between relay state changes (seconds)
INFERENCE_INTERVAL = 0.2  # Run inference every 200ms to reduce load
FRAME_SKIP = 3  # Process only every Nth frame for inference (higher = better FPS but less responsive)
FRAME_WIDTH = 640
FRAME_HEIGHT = 640  # Updated to match model input size for better accuracy
FPS = 30  # Target FPS (updated to match VIDEO_FPS)
DEBUG_MODE = True  # Enable for debugging relay issues
VERBOSE_OUTPUT = False  # Reduce console output
SAVE_EVERY_FRAME = False  # Only save frames with detections to reduce disk I/O
SAVE_INTERVAL = 30  # Only save every 30th frame with detections to improve performance
MAX_SAVED_FRAMES = 20  # Maximum number of frames to keep before overwriting old ones

# Video recording configuration
RECORD_VIDEO = True  # Enable video recording
VIDEO_OUTPUT_DIR = "recordings"  # Directory to save videos
VIDEO_MAX_LENGTH = 600  # Maximum video length in seconds (10 minutes)
VIDEO_CODEC = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4 codec
VIDEO_FPS = 30  # FPS for the recorded video (updated to match record_test_video.py)
VIDEO_RESOLUTION = (640, 640)  # Resolution for the recorded video

# Optimized camera settings for better detection in varying light conditions
CAMERA_SETTINGS = {
    "AeEnable": True,           # Auto exposure
    "AwbEnable": True,          # Auto white balance
    "AeExposureMode": 0,        # Normal exposure mode
    "AeMeteringMode": 0,        # Center-weighted metering
    "ExposureTime": 0,          # Let auto-exposure handle it
    "AnalogueGain": 1.5,        # Increased gain for better low-light performance
    "Brightness": 0.2,          # Increased brightness for better illumination
    "Contrast": 1.1,            # Slightly increased contrast for better visibility
    "Saturation": 1.1,          # Slightly increased saturation for better color
    "FrameRate": VIDEO_FPS,     # Set desired frame rate (using VIDEO_FPS for consistency)
    "AeConstraintMode": 0,      # Normal constraint mode
    "AwbMode": 1,               # Auto white balance mode (1 is typically auto)
    "ExposureValue": 0.5        # Positive EV compensation to improve brightness
}

# Cat class names
CAT_CLASSES = {
    0: "Gary",
    1: "Fred",
    2: "George"  # Updated to match model's label for category_id 2
}

# For COCO dataset
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 
    68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 
    73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 
    78: 'hair drier', 79: 'toothbrush'
}

# Which model to use - set to True to use YOLOv8n COCO model, False for custom cat model
USE_COCO_MODEL = True

# Which classes to detect (only used with COCO model)
# For cats only, use [15]
CLASSES_TO_DETECT = [15]  # Cat class ID in COCO dataset

# Colors for visualization
COLORS = [
    (0, 255, 0),     # Green for class 0
    (255, 0, 0),     # Blue for class 1
    (0, 0, 255),     # Red for class 2
    (255, 255, 0),   # Cyan for class 3
    (255, 0, 255),   # Magenta for class 4
    (0, 255, 255),   # Yellow for class 5
    (128, 0, 0),     # Maroon for class 6
    (0, 128, 0),     # Dark Green for class 7
    (0, 0, 128),     # Navy for class 8
    (128, 128, 0),   # Olive for class 9
    (128, 0, 128),   # Purple for class 10
    (0, 128, 128)    # Teal for class 11
]

# Additional named colors for reference
COLOR_NAMES = {
    'gary': (0, 255, 0),     # Green for Gary
    'george': (255, 0, 0),   # Blue for George
    'fred': (0, 0, 255),     # Red for Fred
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'yellow': (0, 255, 255),
    'red': (0, 0, 255),      # Red for threshold lines and warnings
    'green': (0, 255, 0),    # Green for FPS display
    'unknown': (255, 255, 255)  # White for unknown cats
}

# Global variables
last_squirt_activation = 0
last_action = None
last_action_time = 0
last_valid_overlay = None  # Stores the last valid image overlay from the model

# Cache for relay states to prevent unnecessary toggling
relay_state_cache = {
    RELAY_PINS['squirt']: False,
    RELAY_PINS['left']: False,
    RELAY_PINS['right']: False
}
last_relay_change_time = {
    RELAY_PINS['squirt']: 0,
    RELAY_PINS['left']: 0,
    RELAY_PINS['right']: 0
}

# Sound effects for development mode
SOUND_FILES = [
    'cat1.wav', 'cat2.wav', 'cat3.wav', 'cat4.wav', 'cat5.wav', 'cat6.wav'
]

# Define GPIO pins for relay control
RELAY_PIN_LEFT = RELAY_PINS['left']
RELAY_PIN_RIGHT = RELAY_PINS['right']

def setup_sound():
    """Initialize pygame mixer for sound effects"""
    if DEV_MODE:
        pygame.mixer.init()

def play_sound():
    """Play a random cat sound effect"""
    if DEV_MODE:
        sound_file = random.choice(SOUND_FILES)
        try:
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Error playing sound: {e}")

def setup_camera():
    """Setup camera for capture"""
    if DEV_MODE:
        print("Setting up camera for development mode...")
        camera = cv2.VideoCapture(0)
        
        # Set camera resolution
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        # Check if camera opened successfully
        if not camera.isOpened():
            print("Error: Could not open camera.")
            sys.exit(1)
        
        print(f"Camera initialized with resolution {FRAME_WIDTH}x{FRAME_HEIGHT}")
        return camera
    else:
        print("Setting up Pi camera with optimized detection settings...")
        try:
            from picamera2 import Picamera2
            
            # Create Picamera2 instance
            picam2 = Picamera2()
            
            # Configure camera with improved settings for detection
            camera_config = picam2.create_video_configuration(
                main={
                    "size": (FRAME_WIDTH, FRAME_HEIGHT),
                    "format": "XBGR8888"  # Use XBGR for better color handling
                },
                controls=CAMERA_SETTINGS
            )
            
            # Apply the configuration
            picam2.configure(camera_config)
            
            # Add tuning options to improve low-light performance
            picam2.set_controls({"NoiseReductionMode": 2})  # Enhanced noise reduction
            
            print(f"Pi camera initialized with resolution {FRAME_WIDTH}x{FRAME_HEIGHT}")
            print("Camera settings:")
            for setting, value in CAMERA_SETTINGS.items():
                print(f"  {setting}: {value}")
                
            # Start the camera
            picam2.start()
            
            return picam2
        except Exception as e:
            print(f"Error setting up Pi camera: {e}")
            print("Please check that the camera is properly connected and enabled")
            raise

def setup_gpio():
    """Initialize GPIO pins for relays"""
    if not DEV_MODE:
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Initialize all pins as outputs
            for pin in RELAY_PINS.values():
                GPIO.setup(pin, GPIO.OUT)
            
            # Initialize the squirt relay to OFF state (LOW)
            print(f"Setting SQUIRT relay (pin {RELAY_PINS['squirt']}) to LOW (OFF state)...")
            GPIO.output(RELAY_PINS['squirt'], SQUIRT_RELAY_OFF_STATE)
            
            # Initialize other relays to OFF state (HIGH for active-low relays)
            for name, pin in RELAY_PINS.items():
                if name != 'squirt':
                    print(f"Setting {name} relay (pin {pin}) to HIGH (OFF state for active-low relays)...")
                    GPIO.output(pin, GPIO.HIGH)
            
            print("Successfully initialized GPIO pins")
        except Exception as e:
            print(f"Failed to setup GPIO: {e}")
            raise

def cleanup_camera(camera):
    """Cleanup camera based on mode"""
    if DEV_MODE:
        camera.release()
    else:
        try:
            camera.stop()
            GPIO.cleanup()
        except Exception as e:
            print(f"Error during camera cleanup: {e}")

def get_frame(camera):
    """Get frame from camera based on mode"""
    if DEV_MODE:
        ret, frame = camera.read()
        if not ret:
            raise Exception("Failed to grab frame from webcam")
        # Resize frame to match model input size
        frame = cv2.resize(frame, MODEL_INPUT_SIZE)
        return frame
    else:
        try:
            frame = camera.capture_array()
            
            # Handle different color formats
            if frame.shape[2] == 4:  # If it has 4 channels (XBGR)
                # Convert XBGR to BGR by dropping the X channel
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                # Convert RGB to BGR for OpenCV if needed
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
            # Always resize the frame to the model input size
            frame = cv2.resize(frame, MODEL_INPUT_SIZE)
            return frame
        except Exception as e:
            print(f"Failed to capture frame from Pi camera: {e}")
            raise

def process_detections(frame, results):
    """Process detection results and return detections list"""
    detections = []
    
    if DEV_MODE:
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                score = float(box.conf[0])
                class_id = int(box.cls[0])
                detections.append((x1, y1, x2, y2, score, class_id))
    else:
        # Process Degirum results
        try:
            # YOLO11s model sometimes returns results directly in 'results' and sometimes in 'results.results'
            if hasattr(results, 'results') and results.results:
                result_list = results.results
                if VERBOSE_OUTPUT:
                    print(f"Processing {len(result_list)} detections from results.results")
            elif hasattr(results, 'results') and isinstance(results.results, list):
                result_list = results.results
                if VERBOSE_OUTPUT:
                    print(f"Processing {len(result_list)} detections from results.results list")
            elif isinstance(results, list):
                result_list = results
                if VERBOSE_OUTPUT:
                    print(f"Processing {len(result_list)} detections from direct results list")
            else:
                # No clear list of results found
                result_list = []
                if hasattr(results, '__dict__'):
                    if VERBOSE_OUTPUT:
                        print(f"Results attributes: {list(results.__dict__.keys())}")
            
            # Process detections from the result list
            for detection in result_list:
                try:
                    # Try multiple approaches to extract bounding box information
                    x1, y1, x2, y2, score, class_id = None, None, None, None, None, None
                    
                    # Method 1: Direct dictionary access
                    if isinstance(detection, dict):
                        if 'bbox' in detection:
                            bbox = detection['bbox']
                            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                                x1, y1, x2, y2 = map(int, bbox)
                            
                            score = float(detection.get('score', detection.get('confidence', 0.0)))
                            
                            # Get class ID depending on the model format
                            class_id = int(detection.get('category_id', detection.get('class_id', 0)))
                    
                    # Method 2: Object attribute access
                    elif hasattr(detection, 'bbox'):
                        bbox = detection.bbox
                        
                        # Handle different bbox formats
                        if hasattr(bbox, 'x1') and hasattr(bbox, 'y1') and hasattr(bbox, 'x2') and hasattr(bbox, 'y2'):
                            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
                        elif hasattr(bbox, '__getitem__') and len(bbox) == 4:
                            x1, y1, x2, y2 = map(int, bbox)
                        
                        # Get score and class ID
                        if hasattr(detection, 'score'):
                            score = float(detection.score)
                        elif hasattr(detection, 'confidence'):
                            score = float(detection.confidence)
                        
                        if hasattr(detection, 'class_id'):
                            class_id = int(detection.class_id)
                        elif hasattr(detection, 'category_id'):
                            class_id = int(detection.category_id)
                    
                    # Skip invalid or incomplete detections
                    if None in (x1, y1, x2, y2, score, class_id):
                        continue
                    
                    # For COCO model, filter for cat class only
                    if USE_COCO_MODEL and class_id not in CLASSES_TO_DETECT:
                        continue
                    
                    # Add detection if it meets threshold
                    if score >= DETECTION_THRESHOLD:
                        # Make sure coordinates are within image bounds
                        height, width = frame.shape[:2]
                        x1 = max(0, min(width-1, x1))
                        y1 = max(0, min(height-1, y1))
                        x2 = max(0, min(width-1, x2))
                        y2 = max(0, min(height-1, y2))
                        
                        # Only add if the box has reasonable size
                        if x2 > x1 and y2 > y1 and (x2-x1)*(y2-y1) > 100:  # Minimum area of 100 pixels
                            detections.append((x1, y1, x2, y2, score, class_id))
                            if VERBOSE_OUTPUT:
                                print(f"Added detection: bbox={x1},{y1},{x2},{y2}, score={score:.2f}, class={class_id}")
                
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"Error processing detection: {e}")
            
            # If we have no detections but have a model overlay image, store it for reference
            if len(detections) == 0 and hasattr(results, 'image_overlay') and results.image_overlay is not None:
                # Store the overlay for display
                global last_valid_overlay
                last_valid_overlay = results.image_overlay
                
                if VERBOSE_OUTPUT:
                    print("No detections found above threshold")
        
        except Exception as e:
            if DEBUG_MODE:
                print(f"Error processing detection results: {e}")
                import traceback
                traceback.print_exc()
    
    if VERBOSE_OUTPUT:
        print(f"Returning {len(detections)} processed detections")
    return detections

def activate_relay(pin, duration=0.1):
    """Activate a relay for the specified duration"""
    global last_action, last_action_time
    
    current_time = time.time()
    if not DEV_MODE:
        # Use the set_relay function to correctly handle different relay behaviors
        set_relay(pin, True)  # Turn ON
        time.sleep(duration)
        set_relay(pin, False)  # Turn OFF
    else:
        # Simulate relay activation with messages and sound
        if pin == RELAY_PINS['squirt']:
            last_action = "SQUIRT!"
            print("Squirt!")
            play_sound()
        elif pin == RELAY_PINS['left']:
            last_action = "MOVE LEFT!"
            print("Move Left!")
        elif pin == RELAY_PINS['right']:
            last_action = "MOVE RIGHT!"
            print("Move Right!")
        last_action_time = current_time

def handle_detection(bbox, frame_width):
    """Handle detection and activate appropriate relays"""
    global last_squirt_activation
    
    x1, y1, x2, y2 = map(int, bbox)
    center_x = (x1 + x2) / 2
    relative_position = (center_x / frame_width) - 0.5  # -0.5 to 0.5 range
    
    current_time = time.time()
    
    # Check if we can activate the squirt relay (cooldown period)
    if current_time - last_squirt_activation >= RELAY_SQUIRT_COOLDOWN:
        activate_relay(RELAY_PINS['squirt'], RELAY_SQUIRT_DURATION)
        last_squirt_activation = current_time
    
    # Activate left or right relay based on position
    if relative_position < -CENTER_THRESHOLD:
        activate_relay(RELAY_PINS['left'])
    elif relative_position > CENTER_THRESHOLD:
        activate_relay(RELAY_PINS['right'])
    
    return relative_position

def draw_overlay(frame, relative_position=None):
    """Draw visual indicators on the frame"""
    height, width = frame.shape[:2]
    center_x = width // 2
    
    # Draw center line
    cv2.line(frame, (center_x, 0), (center_x, height), COLORS['yellow'], 1)
    
    # Draw threshold lines
    left_threshold = int(width * (0.5 - CENTER_THRESHOLD))
    right_threshold = int(width * (0.5 + CENTER_THRESHOLD))
    cv2.line(frame, (left_threshold, 0), (left_threshold, height), COLORS['red'], 1)
    cv2.line(frame, (right_threshold, 0), (right_threshold, height), COLORS['red'], 1)
    
    # Add debug information
    if relative_position is not None:
        pos_text = f"Position: {relative_position:.2f}"
        cv2.putText(frame, pos_text, (10, height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['white'], 2)
    
    # Show last action if within last 2 seconds
    if last_action and time.time() - last_action_time < 2:
        cv2.putText(frame, last_action, (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS['red'], 2)

def load_model():
    """Load the YOLO11s model for detection."""
    try:
        print(f"Loading YOLO11s model: {model_name}")
        model_to_load = model_name
        zoo_path = zoo_url
        
        import degirum as dg
        
        # Check if model zoo path exists
        if not os.path.exists(zoo_path):
            print(f"ERROR: Model zoo path not found: {zoo_path}")
            print("Please check that the path is correct and the directory exists.")
            return None
            
        print(f"Model zoo path verified: {zoo_path}")
        
        # Check if the specific model path exists
        specific_model_path = os.path.join(zoo_path, model_to_load)
        if os.path.exists(specific_model_path):
            print(f"Found model at: {specific_model_path}")
        else:
            print(f"WARNING: Specific model path not found: {specific_model_path}")
            
            # List all available models
            try:
                available_models = [f for f in os.listdir(zoo_path) if f.endswith(".hef") or os.path.isdir(os.path.join(zoo_path, f))]
                print(f"Available models in {zoo_path}:")
                for model in available_models:
                    print(f"  - {model}")
            except Exception as e:
                print(f"Warning: Could not list contents of model zoo directory: {e}")
        
        # Load the model with optimized parameters
        model = dg.load_model(
            model_name=model_to_load,
            inference_host_address=inference_host_address,
            zoo_url=zoo_path,
            output_confidence_threshold=DETECTION_THRESHOLD,
            overlay_line_width=3,  # Thicker lines for better visibility
            overlay_font_scale=2.0,  # Font scale for overlay
            overlay_show_probabilities=True,  # Show confidence scores
            output_format="NHWC",  # Use NHWC format for better compatibility
            output_class_activation_threshold=DETECTION_THRESHOLD,  # Make sure class activation threshold matches confidence
            batch_size=1  # Set batch size to 1 for consistent performance
        )
        
        print(f"Model loaded successfully with confidence threshold: {DETECTION_THRESHOLD}")
        
        return model
        
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        print("Detailed error traceback:")
        import traceback
        traceback.print_exc()
        return None

def load_fallback_model():
    """Try to load a generic model just to test if DeGirum is working"""
    try:
        print("Attempting to load a fallback model...")
        import degirum as dg
        
        # Use the local zoo path
        zoo_path = "/home/pi5/degirum_model_zoo"
        
        # Try to list all models in the directory
        try:
            print(f"Looking for fallback models in: {zoo_path}")
            available_models = os.listdir(zoo_path)
            print(f"Available models: {available_models}")
            
            # Filter for .hef models or other known model formats
            potential_models = []
            for model_name in available_models:
                # Add any model that looks like a compiled model
                if model_name != "yolov8n_coco--640x640_quant_hailort_hailo8l_1":  # Skip the main model that failed
                    potential_models.append(model_name)
            
            # Try each potential model
            for model_name in potential_models:
                try:
                    print(f"Trying to load fallback model: {model_name}")
                    model = dg.load_model(
                        model_name=model_name,
                        inference_host_address=inference_host_address,
                        zoo_url=zoo_path,
                        output_confidence_threshold=DETECTION_THRESHOLD
                        # No overlay parameters to avoid compatibility issues
                    )
                    print(f"Successfully loaded fallback model: {model_name}")
                    return model
                except Exception as e:
                    print(f"Failed to load {model_name}: {e}")
                    continue
        except Exception as e:
            print(f"Failed to list models in directory: {e}")
        
        # If local models fail, try models from the public repository
        generic_models = [
            "yolov8n", 
            "yolov8n_coco", 
            "yolov5n_coco",
            "yolov7_tiny_coco",
            "mobilenet_v2_ssd_coco"
        ]
        
        print("Trying models from the public repository...")
        for generic_model in generic_models:
            try:
                print(f"Trying to load generic model: {generic_model}")
                model = dg.load_model(
                    model_name=generic_model,
                    inference_host_address=inference_host_address,
                    zoo_url="degirum/public",  # Use public model zoo
                    output_confidence_threshold=DETECTION_THRESHOLD
                    # No overlay parameters to avoid compatibility issues
                )
                print(f"Successfully loaded fallback model: {generic_model}")
                return model
            except Exception as e:
                print(f"Failed to load {generic_model}: {e}")
                continue
        
        print("Could not load any fallback models")
        return None
    except Exception as e:
        print(f"Error in load_fallback_model: {e}")
        return None

def signal_handler(sig, frame):
    """Handle Ctrl+C signal"""
    print("\nProgram terminated by user")
    if 'camera' in locals():
        cleanup_camera(camera)
    if DEV_MODE:
        pygame.mixer.quit()
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

def save_frame(frame, filename="latest_frame.jpg"):
    """Save a frame to disk for debugging"""
    try:
        # Make sure the image is BGR for OpenCV
        if isinstance(frame, np.ndarray):
            # Create a copy to avoid modifying the original
            save_img = frame.copy()
            
            # Check if the image has the right format
            if save_img.ndim != 3 or save_img.shape[2] != 3:
                print(f"Warning: Unexpected image format: shape={save_img.shape}")
                if save_img.ndim == 2:  # Convert grayscale to BGR
                    save_img = cv2.cvtColor(save_img, cv2.COLOR_GRAY2BGR)
            
            # Verify that image contains valid data
            if np.isnan(save_img).any() or np.isinf(save_img).any():
                print("Warning: Image contains NaN or Inf values, fixing...")
                save_img = np.nan_to_num(save_img)
            
            # Ensure image values are in valid range for uint8
            if save_img.dtype != np.uint8:
                if save_img.max() <= 1.0:
                    # Convert from [0,1] float to [0,255] uint8
                    save_img = (save_img * 255).astype(np.uint8)
                else:
                    # Otherwise just convert to uint8
                    save_img = save_img.astype(np.uint8)
            
            # Save the image
            success = cv2.imwrite(filename, save_img)
            if success:
                print(f"Saved frame to {filename}")
                return True
            else:
                print(f"Failed to save image to {filename}")
                return False
        else:
            print(f"Error: frame is not a valid image: {type(frame)}")
            return False
    except Exception as e:
        print(f"Error saving frame: {e}")
        import traceback
        traceback.print_exc()
        return False

def draw_detection_on_frame(frame, detection):
    """Draw a single detection on the frame with proper formatting"""
    try:
        # Ensure we have a writable copy
        draw_frame = frame.copy()
        
        x1, y1, x2, y2, score, class_id = detection
        
        # Get class name
        if USE_COCO_MODEL:
            class_name = COCO_CLASSES.get(class_id, f"Unknown class {class_id}")
        else:
            class_name = CAT_CLASSES.get(class_id, f"Unknown class {class_id}")
        
        # Get color based on class ID
        color = COLORS[class_id % len(COLORS)]
        
        # Ensure coordinates are valid integers
        height, width = draw_frame.shape[:2]
        x1 = max(0, min(int(x1), width-1))
        y1 = max(0, min(int(y1), height-1))
        x2 = max(0, min(int(x2), width-1))
        y2 = max(0, min(int(y2), height-1))
        
        # Draw with extra thick bounding box to be easily visible
        thickness = 3
        cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, thickness)
        
        # Add filled background for text
        label_text = f"{class_name}: {score:.2f}"
        font_scale = 0.8  # Increased font size
        text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        text_w, text_h = text_size
        
        # Draw filled rectangle for text background - make it more visible
        bg_color = color  # Use same color as box for background
        cv2.rectangle(draw_frame, 
                     (x1, y1 - text_h - 10), 
                     (x1 + text_w + 10, y1), 
                     bg_color, -1)  # -1 means filled
        
        # Draw text with contrasting color (white works well on most colors)
        cv2.putText(draw_frame, label_text, 
                   (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        
        # Also draw a bright crosshair at center of object for visibility
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        crosshair_size = 10
        
        # Draw crosshair lines
        cv2.line(draw_frame, (center_x - crosshair_size, center_y), (center_x + crosshair_size, center_y), (0, 255, 255), 2)
        cv2.line(draw_frame, (center_x, center_y - crosshair_size), (center_x, center_y + crosshair_size), (0, 255, 255), 2)
        
        return draw_frame
    except Exception as e:
        print(f"Error drawing detection: {e}")
        import traceback
        traceback.print_exc()
        return frame  # Return original frame if drawing fails

def test_detection_on_sample_image():
    """
    Test the detection on a sample image to verify model is working.
    This is useful for debugging whether the issue is with the model or the camera.
    """
    try:
        # Load a sample image if available
        sample_image_path = "sample_cat.jpg"
        if os.path.exists(sample_image_path):
            print(f"Testing detection on sample image: {sample_image_path}")
            sample_image = cv2.imread(sample_image_path)
            
            # Resize to match model input size
            sample_image = cv2.resize(sample_image, MODEL_INPUT_SIZE)
            
            # Run inference on the sample image
            import degirum as dg
            model = load_model()  # Load the model
            
            if model:
                try:
                    # Run inference
                    results = model.predict(sample_image)
                    
                    # Save the output
                    if hasattr(results, 'image_overlay') and results.image_overlay is not None:
                        cv2.imwrite("sample_detection.jpg", results.image_overlay)
                        print("Saved sample detection to sample_detection.jpg")
                    
                    # Print detection results
                    if hasattr(results, 'results'):
                        print(f"Sample image results: {len(results.results)} detections")
                    
                    return True
                except Exception as e:
                    print(f"Error running inference on sample: {e}")
            
            return False
    except Exception as e:
        print(f"Error in test detection: {e}")
        return False
    
    return False  # No sample image found

def test_degirum_setup():
    """Test if DeGirum and Hailo are correctly set up"""
    try:
        import degirum as dg
        print("DeGirum package is installed.")
        
        # Try to get DeGirum version
        if hasattr(dg, '__version__'):
            print(f"DeGirum version: {dg.__version__}")
        
        # Test if Hailo runtime is available
        try:
            # List available devices if possible
            if hasattr(dg, 'list_devices'):
                devices = dg.list_devices()
                print(f"Available devices: {devices}")
            else:
                print("DeGirum does not have list_devices method.")
                
            # Alternative way to check for Hailo
            import subprocess
            try:
                result = subprocess.run(['hailortcli', 'device', 'show'], 
                                        capture_output=True, text=True, timeout=5)
                print("Hailo device information:")
                print(result.stdout)
                if result.returncode != 0:
                    print(f"Warning: hailortcli returned non-zero exit code: {result.returncode}")
                    print(f"Error output: {result.stderr}")
            except Exception as e:
                print(f"Could not run hailortcli: {e}")
            
            return True
        except Exception as e:
            print(f"Error checking Hailo runtime: {e}")
            return False
            
    except ImportError:
        print("ERROR: DeGirum package is not installed or cannot be imported.")
        print("Please install DeGirum package using:")
        print("pip install degirum")
        return False
    except Exception as e:
        print(f"Error testing DeGirum setup: {e}")
        return False

class FPSCounter:
    """Class to calculate FPS"""
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0
        
    def get_fps(self):
        """Calculate FPS based on frame count and elapsed time"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        # Update FPS calculation every second
        if elapsed_time >= 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
            
        return self.fps

def cleanup_old_frames():
    """Clean up old saved frames to prevent filling the disk"""
    try:
        # Get all jpg files in the current directory
        jpg_files = [f for f in os.listdir('.') if f.startswith('detection_') and f.endswith('.jpg')]
        
        # Sort by modification time (newest first)
        jpg_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        
        # If we have more than MAX_SAVED_FRAMES, remove the oldest ones
        if len(jpg_files) > MAX_SAVED_FRAMES:
            files_to_remove = jpg_files[MAX_SAVED_FRAMES:]
            for old_file in files_to_remove:
                try:
                    os.remove(old_file)
                    if DEBUG_MODE:
                        print(f"Removed old frame: {old_file}")
                except Exception as e:
                    print(f"Error removing old frame {old_file}: {e}")
                    
        # Also clean up old FPS report images
        fps_files = [f for f in os.listdir('.') if f.startswith('fps_report_') and f.endswith('.jpg')]
        fps_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        
        # Keep only the 5 most recent FPS reports
        if len(fps_files) > 5:
            for old_file in fps_files[5:]:
                try:
                    os.remove(old_file)
                except Exception:
                    pass
    except Exception as e:
        if DEBUG_MODE:
            print(f"Error cleaning up old frames: {e}")

def init_video_writer():
    """Initialize video writer for recording"""
    if not RECORD_VIDEO:
        return None
        
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(VIDEO_OUTPUT_DIR):
            os.makedirs(VIDEO_OUTPUT_DIR)
            print(f"Created video output directory: {VIDEO_OUTPUT_DIR}")
        
        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = os.path.join(VIDEO_OUTPUT_DIR, f"cat_detector_{timestamp}.mp4")
        
        # Create video writer
        video_writer = cv2.VideoWriter(
            video_filename, 
            VIDEO_CODEC, 
            VIDEO_FPS, 
            VIDEO_RESOLUTION
        )
        
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer. Check codec and path: {video_filename}")
            return None
            
        print(f"Video recording initialized: {video_filename}")
        return video_writer
    except Exception as e:
        print(f"Error initializing video writer: {e}")
        return None

def cleanup_video_writer(video_writer):
    """Clean up video writer"""
    if video_writer is not None:
        try:
            video_writer.release()
            print("Video writer released")
        except Exception as e:
            print(f"Error releasing video writer: {e}")

def create_new_video_writer(video_writer):
    """Create a new video writer when the current video reaches max length"""
    # First close the current writer
    if video_writer is not None:
        cleanup_video_writer(video_writer)
    
    # Then create a new one
    return init_video_writer()

def draw_relay_status(frame, active_relays):
    """Draw relay status indicators on the frame"""
    height, width = frame.shape[:2]
    
    # Draw relay status in top-right corner
    cv2.rectangle(frame, (width - 180, 80), (width, 200), (0, 0, 0), -1)
    
    y_offset = 100
    
    # Draw center/squirt relay status
    center_status = "SQUIRT: ON" if active_relays.get('squirt', False) else "SQUIRT: OFF"
    color = (0, 255, 0) if active_relays.get('squirt', False) else (0, 0, 255)
    cv2.putText(frame, center_status, (width - 170, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw left relay status
    y_offset += 30
    left_status = "LEFT: ON" if active_relays.get('left', False) else "LEFT: OFF"
    color = (0, 255, 0) if active_relays.get('left', False) else (0, 0, 255)
    cv2.putText(frame, left_status, (width - 170, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw right relay status
    y_offset += 30
    right_status = "RIGHT: ON" if active_relays.get('right', False) else "RIGHT: OFF"
    color = (0, 255, 0) if active_relays.get('right', False) else (0, 0, 255)
    cv2.putText(frame, right_status, (width - 170, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame

def process_actions(frame, detections, fps):
    """Process detections and take appropriate actions"""
    frame_width = frame.shape[1]
    
    # Create a copy for drawing
    annotated_frame = frame.copy()
    
    # Add visual guides
    height, width = annotated_frame.shape[:2]
    center_x = width // 2
    
    # Draw center line
    cv2.line(annotated_frame, (center_x, 0), (center_x, height), (0, 255, 255), 2)
    
    # Draw threshold lines
    left_threshold = int(width * (0.5 - CENTER_THRESHOLD))
    right_threshold = int(width * (0.5 + CENTER_THRESHOLD))
    cv2.line(annotated_frame, (left_threshold, 0), (left_threshold, height), (0, 0, 255), 2)
    cv2.line(annotated_frame, (right_threshold, 0), (right_threshold, height), (0, 0, 255), 2)
    
    # Add header with FPS
    fps_text = f"FPS: {fps:.1f}"
    cv2.rectangle(annotated_frame, (0, 0), (200, 40), (0, 0, 0), -1)
    cv2.putText(annotated_frame, fps_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Add model info
    model_text = f"Model: YOLO11s"
    threshold_text = f"Threshold: {DETECTION_THRESHOLD:.2f}"
    cv2.rectangle(annotated_frame, (width - 250, 0), (width, 70), (0, 0, 0), -1)
    cv2.putText(annotated_frame, model_text, (width - 240, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(annotated_frame, threshold_text, (width - 240, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Track active relays for display
    active_relays = {
        'squirt': False,
        'left': False,
        'right': False
    }
    
    # Frame counter for limiting saved images
    frame_counter = int(time.time()) % SAVE_INTERVAL
    
    # Clean up old frames periodically
    if frame_counter == 0:
        cleanup_old_frames()
    
    # Process detections if we have any
    if len(detections) > 0:
        if VERBOSE_OUTPUT:
            print(f"Detected {len(detections)} objects")
        
        # Save detection frame periodically
        if frame_counter == 0:
            detection_filename = f"detection_{int(time.time())}.jpg"
        
        # We'll find the primary detection to use for relay control
        # If multiple cats are detected, use the one with highest confidence
        primary_detection = max(detections, key=lambda d: d[4])  # d[4] is the confidence
        x1, y1, x2, y2, score, class_id = primary_detection
        
        # Calculate relative position for GPIO control
        object_center_x = (x1 + x2) / 2
        relative_position = (object_center_x / frame_width) - 0.5
        
        # Set relay states based on object position
        if not DEV_MODE:
            # Always activate squirt relay for any detection
            active_relays['squirt'] = True
            set_relay(RELAY_PINS['squirt'], True)
            
            if relative_position < -CENTER_THRESHOLD:
                # Object is on the left side
                active_relays['left'] = True
                if DEBUG_MODE and set_relay(RELAY_PIN_LEFT, True):
                    print("Cat on LEFT side - activating LEFT relay")
                set_relay(RELAY_PIN_RIGHT, False)
            elif relative_position > CENTER_THRESHOLD:
                # Object is on the right side
                active_relays['right'] = True
                set_relay(RELAY_PIN_LEFT, False)
                if DEBUG_MODE and set_relay(RELAY_PIN_RIGHT, True):
                    print("Cat on RIGHT side - activating RIGHT relay")
            else:
                # Object is centered
                set_relay(RELAY_PIN_LEFT, False)
                set_relay(RELAY_PIN_RIGHT, False)
                if DEBUG_MODE:
                    print("Cat in CENTER - directional relays OFF")
        
        # Process each detection for display
        for detection in detections:
            x1, y1, x2, y2, score, class_id = detection
            
            # Get class name based on which model we're using
            if USE_COCO_MODEL:
                class_name = COCO_CLASSES.get(class_id, f"Unknown class {class_id}")
            else:
                class_name = CAT_CLASSES.get(class_id, f"Unknown class {class_id}")
            
            if VERBOSE_OUTPUT:
                print(f"- {class_name}: confidence={score:.2f}, box=({x1},{y1},{x2},{y2})")
            
            # Draw detection on the annotated frame
            color_index = class_id % len(COLORS)
            color = COLORS[color_index]
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label with confidence
            label = f"{class_name}: {score:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1-30), (x1+len(label)*15, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw directional indicators based on primary detection position
        if relative_position < -CENTER_THRESHOLD:
            # Left indicator
            direction_text = "MOVE LEFT"
            cv2.putText(annotated_frame, direction_text, (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            # Draw left arrow
            arrow_start = (width // 4, height - 50)
            arrow_end = (width // 8, height - 50)
            cv2.arrowedLine(annotated_frame, arrow_start, arrow_end, (0, 0, 255), 4, tipLength=0.5)
        elif relative_position > CENTER_THRESHOLD:
            # Right indicator
            direction_text = "MOVE RIGHT"
            cv2.putText(annotated_frame, direction_text, (width - 240, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            # Draw right arrow
            arrow_start = (width // 4 * 3, height - 50)
            arrow_end = (width // 8 * 7, height - 50)
            cv2.arrowedLine(annotated_frame, arrow_start, arrow_end, (0, 0, 255), 4, tipLength=0.5)
        else:
            # Center indicator
            direction_text = "CENTER"
            cv2.putText(annotated_frame, direction_text, (width // 2 - 80, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Add position value
        pos_text = f"Position: {relative_position:.2f}"
        cv2.putText(annotated_frame, pos_text, (10, height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        # Save frame if it's time
        if frame_counter == 0:
            # Save the annotated frame
            cv2.imwrite(detection_filename, annotated_frame)
            if DEBUG_MODE:
                print(f"Saved annotated detection image to {detection_filename}")
    else:
        # No detections found, turn off all relays
        if not DEV_MODE:
            # Turn off relays when no detections are found
            # Only print debug messages if the relay state actually changes
            if set_relay(RELAY_PINS['squirt'], False) and DEBUG_MODE:
                print("No detections - squirt relay OFF")
            if set_relay(RELAY_PIN_LEFT, False) and DEBUG_MODE:
                print("No detections - left relay OFF")
            if set_relay(RELAY_PIN_RIGHT, False) and DEBUG_MODE:
                print("No detections - right relay OFF")
    
    # Add timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(annotated_frame, timestamp, (width - 250, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw relay status on frame
    annotated_frame = draw_relay_status(annotated_frame, active_relays)
    
    return annotated_frame

def test_model_on_sample(model):
    """Test the model on a sample image"""
    try:
        # Load sample image
        sample_path = "sample_cat.jpg"
        if not os.path.exists(sample_path):
            print(f"Sample image not found: {sample_path}")
            return
            
        print(f"Loading sample image: {sample_path}")
        sample_img = cv2.imread(sample_path)
        
        if sample_img is None:
            print("Failed to load sample image")
            return
            
        # Perform inference on sample image
        print("Running inference on sample image...")
        
        # Run inference
        if DEV_MODE:
            results = model(sample_img, conf=DETECTION_THRESHOLD)
        else:
            results_generator = model.predict_batch([sample_img])
            results = next(results_generator)
        
        print(f"Sample image inference result type: {type(results)}")
        
        # Process detections
        detections = process_detections(sample_img, results)
        
        # Create visualization
        annotated_img = sample_img.copy()
        
        # Draw detections
        if len(detections) > 0:
            print(f"Found {len(detections)} objects in sample image:")
            
            for detection in detections:
                x1, y1, x2, y2, score, class_id = detection
                
                # Get class name
                if USE_COCO_MODEL:
                    class_name = COCO_CLASSES.get(class_id, f"Unknown class {class_id}")
                else:
                    class_name = CAT_CLASSES.get(class_id, f"Unknown class {class_id}")
                
                print(f"- {class_name}: confidence={score:.2f}, box=({x1},{y1},{x2},{y2})")
                
                # Draw bounding box
                color = COLORS[class_id % len(COLORS)]
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                
                # Add label with confidence
                label = f"{class_name}: {score:.2f}"
                cv2.rectangle(annotated_img, (x1, y1-30), (x1+len(label)*15, y1), color, -1)
                cv2.putText(annotated_img, label, (x1, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
            # Save the annotated sample image
            cv2.imwrite("sample_detection.jpg", annotated_img)
            print("Saved sample detection results to sample_detection.jpg")
        else:
            print("No objects detected in sample image")
            
            # If we have an image overlay, save it
            if hasattr(results, 'image_overlay') and results.image_overlay is not None:
                print("Saving model overlay for sample image")
                cv2.imwrite("sample_overlay.jpg", results.image_overlay)
                
                # Also save the original for comparison
                cv2.imwrite("sample_original.jpg", sample_img)
                
                # Create side-by-side comparison
                if results.image_overlay.shape == sample_img.shape:
                    comparison = np.hstack((sample_img, results.image_overlay))
                    cv2.imwrite("sample_comparison.jpg", comparison)
    
    except Exception as e:
        print(f"Error testing model on sample image: {e}")
        import traceback
        traceback.print_exc()

def read_frame(camera):
    """Read a frame from the camera"""
    try:
        if DEV_MODE:
            # OpenCV camera in dev mode
            ret, frame = camera.read()
            
            if not ret:
                print("Failed to capture frame")
                return None
                
            return frame
        else:
            # Picamera2 in production mode
            try:
                # Get frame directly from Picamera2
                frame = camera.capture_array()
                
                # Handle different color formats
                if frame.shape[2] == 4:  # If it has 4 channels (XBGR)
                    # Convert XBGR to BGR by dropping the X channel
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                else:
                    # Convert RGB to BGR for OpenCV if needed
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Resize if needed
                if frame.shape[0] != FRAME_HEIGHT or frame.shape[1] != FRAME_WIDTH:
                    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                
                return frame
            except Exception as e:
                print(f"Error capturing frame from Pi camera: {e}")
                return None
    except Exception as e:
        print(f"Error reading frame: {e}")
        return None

def init_gpio():
    """Initialize GPIO pins for relay control"""
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        
        # Setup all relay pins as OUTPUT
        GPIO.setup(RELAY_PIN_LEFT, GPIO.OUT)
        GPIO.setup(RELAY_PIN_RIGHT, GPIO.OUT)
        GPIO.setup(RELAY_PINS['squirt'], GPIO.OUT)
        GPIO.setup(RELAY_PINS['unused'], GPIO.OUT)  # Setup unused relay too
        
        # For normally closed relays with active LOW:
        # - To turn relay OFF (close circuit): Set GPIO HIGH (deactivate relay)
        # - To turn relay ON (open circuit): Set GPIO LOW (activate relay)
        initial_state = False  # We always want relays to start in the OFF state
        
        # Initialize all relays to OFF state
        print("Setting all relays to OFF state...")
        for name, pin in RELAY_PINS.items():
            set_relay(pin, initial_state)
        
        print(f"GPIO initialized with relay pins: squirt={RELAY_PINS['squirt']}, left={RELAY_PIN_LEFT}, right={RELAY_PIN_RIGHT}")
        print(f"Relays are {'ACTIVE LOW' if RELAY_ACTIVE_LOW else 'ACTIVE HIGH'} and {'NORMALLY CLOSED' if RELAY_NORMALLY_CLOSED else 'NORMALLY OPEN'}")
    except ImportError:
        print("WARNING: RPi.GPIO module not available, GPIO control disabled")
    except Exception as e:
        print(f"Error initializing GPIO: {e}")

def set_relay(pin, state):
    """
    Set relay state (True = ON, False = OFF) with state caching and hysteresis
    
    Special handling for squirt relay which has opposite behavior from other relays:
    - Squirt relay: HIGH = ON, LOW = OFF
    - Other relays: LOW = ON, HIGH = OFF (active-low)
    """
    global relay_state_cache, last_relay_change_time
    
    # Get current time for hysteresis check
    current_time = time.time()
    
    # Skip if the state hasn't changed
    if pin in relay_state_cache and relay_state_cache[pin] == state:
        return False
    
    # Apply hysteresis to prevent rapid toggling
    if pin in last_relay_change_time:
        time_since_last_change = current_time - last_relay_change_time[pin]
        if time_since_last_change < RELAY_HYSTERESIS_TIME:
            if DEBUG_MODE:
                print(f"Skipping relay {pin} change due to hysteresis ({time_since_last_change:.2f}s < {RELAY_HYSTERESIS_TIME}s)")
            return False
    
    try:
        import RPi.GPIO as GPIO
        
        # Special handling for squirt relay which has opposite behavior
        if pin == RELAY_PINS['squirt']:
            # Squirt relay: HIGH = ON, LOW = OFF
            gpio_state = SQUIRT_RELAY_ON_STATE if state else SQUIRT_RELAY_OFF_STATE
            if DEBUG_MODE:
                print(f"SQUIRT relay {pin} set to {'ON' if state else 'OFF'} (GPIO {'HIGH' if gpio_state == GPIO.HIGH else 'LOW'})")
        else:
            # Standard relays: active-low behavior (LOW = ON, HIGH = OFF)
            gpio_state = GPIO.LOW if state else GPIO.HIGH
            if DEBUG_MODE:
                print(f"Standard relay {pin} set to {'ON' if state else 'OFF'} (GPIO {'LOW' if gpio_state == GPIO.LOW else 'HIGH'})")
        
        # Set the GPIO pin state
        GPIO.output(pin, gpio_state)
        
        # Update state cache and timestamp
        relay_state_cache[pin] = state
        last_relay_change_time[pin] = current_time
        
        return True
    except (ImportError, NameError):
        # If we can't import GPIO, just print what we would do
        if DEBUG_MODE:
            print(f"Would set relay {pin} to {'ON' if state else 'OFF'}")
        
        # Still update the cache for consistent behavior
        relay_state_cache[pin] = state
        last_relay_change_time[pin] = current_time
        return True
    except Exception as e:
        print(f"Error setting relay {pin}: {e}")
        return False

def cleanup():
    """Clean up resources before exiting"""
    print("Cleaning up resources...")
    
    # Ensure all relays are turned OFF before cleanup
    try:
        if not DEV_MODE:
            print("Turning off all relays...")
            # Use specific handling for the squirt relay to ensure it's OFF
            print(f"Setting squirt relay (pin {RELAY_PINS['squirt']}) to OFF state...")
            GPIO.output(RELAY_PINS['squirt'], SQUIRT_RELAY_OFF_STATE)
            
            # Turn off all other relays
            for name, pin in RELAY_PINS.items():
                if name != 'squirt' and name != 'unused':
                    print(f"Setting {name} relay (pin {pin}) to OFF state...")
                    GPIO.output(pin, GPIO.HIGH)  # HIGH = OFF for standard active-low relays
            
            time.sleep(0.5)  # Give time for states to take effect
            
            # Now clean up GPIO
            GPIO.cleanup()
            print("GPIO pins cleaned up")
    except Exception as e:
        print(f"Error during GPIO cleanup: {e}")
    
    # Release camera if it exists
    try:
        if 'camera' in globals() and camera is not None:
            if DEV_MODE:
                camera.release()
            else:
                camera.stop()
            print("Camera released")
    except Exception as e:
        print(f"Error releasing camera: {e}")
        
    print("Cleanup completed")

def print_config():
    """Print current configuration settings"""
    print("\n=== Configuration ===")
    print(f"Model type: {'COCO YOLOv8n' if USE_COCO_MODEL else 'Custom cat model'}")
    if USE_COCO_MODEL:
        print(f"Classes to detect: {[COCO_CLASSES[i] for i in CLASSES_TO_DETECT]}")
    else:
        print(f"Classes to detect: {list(CAT_CLASSES.values())}")
    print(f"Detection threshold: {DETECTION_THRESHOLD}")
    print(f"Display resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"Target FPS: {FPS}")
    print(f"Debug mode: {'Enabled' if DEBUG_MODE else 'Disabled'}")
    print(f"Development mode: {'Enabled' if DEV_MODE else 'Disabled'}")
    print("=== End Configuration ===\n")
    
def print_header():
    """Print program header information"""
    print("\n===============================================")
    print("=  Cat Detection System with Hailo AI Kit    =")
    print("===============================================")
    print(f"OpenCV version: {cv2.__version__}")
    
    # Check if running in headless mode
    if 'DISPLAY' not in os.environ:
        print("Running in headless mode - terminal output only")
    else:
        print("Running with display support")
        
    # Print platform-specific information
    if sys.platform.startswith('linux'):
        try:
            # Try to import Pi-specific libraries to detect platform
            import RPi.GPIO
            print("Successfully imported Pi-specific modules")
        except ImportError:
            print("Not running on Raspberry Pi (RPi.GPIO not available)")
    else:
        print(f"Running on platform: {sys.platform}")
    print("")

def test_relays():
    """Test all relays in sequence to verify they're working"""
    if DEV_MODE:
        print("Skipping relay test in development mode")
        return
        
    try:
        print("\n==== RELAY TEST ====")
        print("Testing all relays in sequence...")
        
        # First turn all relays OFF to establish baseline
        print("Setting all relays to OFF state...")
        # Special handling for squirt relay
        GPIO.output(RELAY_PINS['squirt'], SQUIRT_RELAY_OFF_STATE)
        print(f"SQUIRT relay (pin {RELAY_PINS['squirt']}) set to {'LOW' if SQUIRT_RELAY_OFF_STATE == GPIO.LOW else 'HIGH'} (OFF)")
        
        # Handle other relays
        for name, pin in RELAY_PINS.items():
            if name != 'squirt' and name != 'unused':
                GPIO.output(pin, GPIO.HIGH)  # Standard relays: HIGH = OFF (active-low)
                print(f"{name.upper()} relay (pin {pin}) set to HIGH (OFF)")
        
        time.sleep(1.0)
        
        # Then test each relay individually
        for name, pin in RELAY_PINS.items():
            if name == 'unused':
                continue
                
            print(f"\nTesting {name.upper()} relay (pin {pin})...")
            
            if name == 'squirt':
                # Turn ON squirt relay (HIGH)
                print(f"  Setting {name.upper()} relay to ON (HIGH)")
                GPIO.output(pin, SQUIRT_RELAY_ON_STATE)
                time.sleep(1.0)
                
                # Turn OFF squirt relay (LOW)
                print(f"  Setting {name.upper()} relay to OFF (LOW)")
                GPIO.output(pin, SQUIRT_RELAY_OFF_STATE)
            else:
                # Turn ON standard relay (LOW)
                print(f"  Setting {name.upper()} relay to ON (LOW)")
                GPIO.output(pin, GPIO.LOW)
                time.sleep(1.0)
                
                # Turn OFF standard relay (HIGH)
                print(f"  Setting {name.upper()} relay to OFF (HIGH)")
                GPIO.output(pin, GPIO.HIGH)
            
            time.sleep(0.5)
        
        # Test all relays at once
        print("\nTesting all relays together...")
        
        # Turn ON all relays
        print("  Setting ALL relays to ON")
        GPIO.output(RELAY_PINS['squirt'], SQUIRT_RELAY_ON_STATE)  # HIGH for squirt
        for name, pin in RELAY_PINS.items():
            if name != 'squirt' and name != 'unused':
                GPIO.output(pin, GPIO.LOW)  # LOW for standard relays
        
        time.sleep(1.5)
        
        # Turn OFF all relays
        print("  Setting ALL relays to OFF")
        GPIO.output(RELAY_PINS['squirt'], SQUIRT_RELAY_OFF_STATE)  # LOW for squirt
        for name, pin in RELAY_PINS.items():
            if name != 'squirt' and name != 'unused':
                GPIO.output(pin, GPIO.HIGH)  # HIGH for standard relays
            
        print("\nRelay test completed")
        print("====================\n")
    except Exception as e:
        print(f"Error testing relays: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    try:
        print_header()
        
        # Test DeGirum and Hailo setup first
        if not DEV_MODE:
            test_degirum_setup()
        
        # Load AI model with proper error handling
        model = load_model()
        
        if model is None:
            print("Primary model loading failed. Trying fallback model...")
            model = load_fallback_model()
            
        if model is None:
            print("ERROR: Failed to load any model. Check the error messages above for details.")
            print("Check that the DeGirum package is installed correctly and Hailo accelerator is connected.")
            print("Exiting program.")
            sys.exit(1)

        # Configure camera
        camera = setup_camera()
        
        # Initialize video writer if recording is enabled
        video_writer = init_video_writer() if RECORD_VIDEO else None
        video_start_time = time.time()
        
        # Initialize GPIO if not in dev mode
        if not DEV_MODE:
            init_gpio()
            # Test all relays
            test_relays()
        
        # Test model on a sample image if it exists
        if os.path.exists("sample_cat.jpg") and not DEV_MODE:
            test_model_on_sample(model)
        
        # Print configuration
        print_config()
        
        # Initialize variables for smoothing detections
        last_valid_detections = []
        no_detection_frames = 0
        max_no_detection_frames = 3  # Keep using last detection for 3 frames
        
        # Track performance metrics
        fps_counter = FPSCounter()
        frame_count = 0
        last_fps_print = time.time()
        inference_count = 0
        
        # Main loop
        while True:
            loop_start_time = time.time()
            
            # Read frame from camera
            frame = read_frame(camera)
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            # We always increment frame count and calculate FPS
            frame_count += 1
            fps = fps_counter.get_fps()
            
            # Determine if we should run inference on this frame
            should_run_inference = (frame_count % FRAME_SKIP == 0)
            
            # Run inference if needed
            if should_run_inference:
                inference_count += 1
                inference_start_time = time.time()
                
                if DEBUG_MODE and inference_count % 10 == 0:
                    print(f"Running inference #{inference_count} at t={inference_start_time:.2f}s")
                
                # Perform inference
                try:
                    if DEV_MODE:
                        results = model(frame, conf=DETECTION_THRESHOLD)
                    else:
                        # Use Degirum's predict_batch method
                        results_generator = model.predict_batch([frame])
                        results = next(results_generator)
                    
                    # Process detection results
                    detections = process_detections(frame, results)
                    
                    # Apply detection smoothing
                    if not detections:
                        no_detection_frames += 1
                        if no_detection_frames <= max_no_detection_frames and last_valid_detections:
                            if DEBUG_MODE and no_detection_frames == 1:
                                print(f"No detections in this frame, using last valid detection")
                            detections = last_valid_detections
                        else:
                            if DEBUG_MODE and no_detection_frames == max_no_detection_frames + 1:
                                print("No recent valid detections to use")
                            # Clear detections completely
                            detections = []
                    else:
                        # We have valid detections, reset counter and save for future use
                        no_detection_frames = 0
                        last_valid_detections = detections.copy()
                        
                    inference_time = time.time() - inference_start_time
                    if DEBUG_MODE and inference_count % 10 == 0:
                        print(f"Inference took {inference_time*1000:.1f}ms")
                except Exception as e:
                    print(f"Error during inference: {e}")
                    detections = last_valid_detections if no_detection_frames <= max_no_detection_frames else []
            else:
                # Reuse last detections when skipping inference
                detections = last_valid_detections if no_detection_frames <= max_no_detection_frames else []
            
            # Process actions and update display regardless of whether we did inference
            processed_frame = process_actions(frame, detections, fps)
            
            # Record video if enabled
            if RECORD_VIDEO and video_writer is not None:
                try:
                    # Write the processed frame (with annotations) to video
                    video_writer.write(processed_frame)
                    
                    # Check if we need to create a new video file (max length reached)
                    video_duration = time.time() - video_start_time
                    if video_duration >= VIDEO_MAX_LENGTH:
                        print(f"Video reached maximum length ({VIDEO_MAX_LENGTH}s), creating new file")
                        video_writer = create_new_video_writer(video_writer)
                        video_start_time = time.time()
                        
                except Exception as e:
                    print(f"Error writing video frame: {e}")
            
            # Only save FPS report periodically
            current_time = time.time()
            if current_time - last_fps_print >= 10.0:  # Every 10 seconds
                print(f"Current FPS: {fps:.1f}")
                cv2.imwrite(f"fps_report_{int(current_time)}.jpg", processed_frame)
                last_fps_print = current_time
            
            # Sleep to maintain target FPS
            frame_time = time.time() - loop_start_time
            sleep_time = max(0, (1.0 / FPS) - frame_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting...")
        
        # Make sure to turn off all relays when exiting
        if not DEV_MODE:
            set_relay(RELAY_PIN_LEFT, False)
            set_relay(RELAY_PIN_RIGHT, False)
            set_relay(RELAY_PINS['squirt'], False)
        
    except Exception as e:
        print(f"Error in main loop: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up resources
        cleanup()
        
        # Clean up video writer
        if RECORD_VIDEO and 'video_writer' in locals() and video_writer is not None:
            cleanup_video_writer(video_writer)
            
        print("Program terminated")

try:
    main()
except Exception as e:
    print(f"Error: {e}")
    if 'camera' in locals():
        cleanup_camera(camera)
    if DEV_MODE:
        pygame.mixer.quit()
    
print("Resources cleaned up")