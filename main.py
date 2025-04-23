# main_pd.py
# This project detects cats/dogs and triggers a water gun to squirt them.
# It runs on a Raspberry Pi 5 with a Hailo AI Kit and a four-relay HAT.
# Uses PD control for smoother centering.

import os
import cv2
import time
import numpy as np
import random
import datetime
import signal
import sys
import traceback
import RPi.GPIO as GPIO
from picamera2 import Picamera2
import degirum as dg
import threading
import select

# Print the version of OpenCV being used
print(f"OpenCV version: {cv2.__version__}")
try:
    import pygame
    SOUND_ENABLED = True
    print("Pygame imported successfully for sound.")
except ImportError:
    SOUND_ENABLED = False
    print("Pygame not found, sound effects disabled.")

# --- Configuration Constants ---

# GPIO Pin Setup (BCM Mode)
RELAY_PINS = {
    'squirt': 5,  # Squirt relay (triggers water gun)
    'left': 13,   # Left relay - controls LEFT movement
    'right': 6,   # Right relay - controls RIGHT movement
    'unused': 15  # Unused relay
}
RELAY_ACTIVE_LOW = True # True if relay activates on LOW signal
# Squirt relay behavior differs from movement relays
SQUIRT_RELAY_ON_STATE = GPIO.HIGH  # GPIO state to turn SQUIRT relay ON
SQUIRT_RELAY_OFF_STATE = GPIO.LOW  # GPIO state to turn SQUIRT relay OFF

# Model Setup
HAILO_ZOO_PATH = "/home/pi5/degirum_model_zoo" # Local path to DeGirum model zoo
HAILO_MODEL_NAME = "yolov8s_coco--640x640_quant_hailort_hailo8l_1" # Example model - verify yours
# HAILO_MODEL_NAME = "yolo11s_silu_coco--640x640_quant_hailort_hailo8l_1" # Original model name
HAILO_INFERENCE_ADDRESS = "@local" # Use local Hailo accelerator

# Detection & Classes
DETECTION_THRESHOLD = 0.30 # Confidence threshold
MODEL_INPUT_SIZE = (640, 640) # Must match model input
# COCO Class ID for 'cat' is 15, 'dog' is 16
CLASSES_TO_DETECT = [15, 16] # Detect cats and dogs

# PD Control Parameters (REQUIRES TUNING)
PD_KP = 0.6                 # Proportional gain
PD_KD = 0.1                 # Derivative gain
PD_CENTER_THRESHOLD = 0.05  # Dead zone around center (+/- %)
PD_MIN_PULSE = 0.01         # Minimum effective pulse duration (seconds)
PD_MAX_PULSE = 0.15         # Maximum pulse duration (seconds)
PD_MOVEMENT_COOLDOWN = 0.1  # Cooldown between movements (seconds)

# Timing & Performance
RELAY_SQUIRT_DURATION = 0.2  # Duration for squirt relay activation (seconds)
RELAY_SQUIRT_COOLDOWN = 1.0  # Cooldown between squirts (seconds)
FRAME_SKIP = 2               # Run inference every Nth frame (1 = every frame)
TARGET_FPS = 30              # Target camera capture/processing FPS
POSITION_RESET_TIME = 10.0   # Reset PD state if no detection for this long (seconds)

# Camera Settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 640
CAMERA_SETTINGS = {
    "AeEnable": True, "AwbEnable": True, "AeExposureMode": 0,
    "AeMeteringMode": 0, "ExposureTime": 0, "AnalogueGain": 1.5,
    "Brightness": 0.2, "Contrast": 1.1, "Saturation": 1.1,
    "FrameRate": TARGET_FPS, "AeConstraintMode": 0, "AwbMode": 1,
    "ExposureValue": 0.5, "NoiseReductionMode": 2
}

# Video Recording
RECORD_VIDEO = True
VIDEO_OUTPUT_DIR = "recordings"
VIDEO_MAX_LENGTH = 600 # Seconds (10 minutes)
VIDEO_CODEC = cv2.VideoWriter_fourcc(*'mp4v')
VIDEO_FPS = TARGET_FPS # Match camera FPS
VIDEO_RESOLUTION = (FRAME_WIDTH, FRAME_HEIGHT)
VIDEO_START_RECORD_DURATION = 60 # Record for this many seconds after script start, even if RECORD_VIDEO is False

# Debugging & Output
DEBUG_MODE = True     # Master debug switch
VERBOSE_OUTPUT = True # Print detailed logs
SAVE_DEBUG_FRAMES = True # Save frames with detections periodically
DEBUG_FRAME_INTERVAL = 10 # Save every Nth detected frame
MAX_SAVED_FRAMES = 20 # Max number of debug frames to keep

# COCO Class Names (for labels)
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
# Colors for bounding boxes (cycles through)
COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128)
]
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)

# Sound effects (optional)
SOUND_FILES = ['cat1.wav', 'cat2.wav', 'cat3.wav', 'cat4.wav', 'cat5.wav', 'cat6.wav']


# --- Global State Variables ---
last_squirt_activation = 0.0
last_action = "Initializing"
last_action_time = 0.0
last_detection_time = 0.0 # Time of the last valid detection
# PD Control State
previous_error = 0.0        # Store the last error for derivative calculation
last_movement_time = 0.0    # Time the last movement command was issued
# Other State
video_writer = None
camera = None
model = None
frame_num_global = 0 # Keep track of frames for periodic saving
# New variable for controlling program execution
running = True

# --- Terminal Input Functions ---

def print_current_settings():
    """Print the current PD settings to the console."""
    print("\n--- Current PD Control Settings ---")
    print(f"PD_KP = {PD_KP:.2f}")
    print(f"PD_KD = {PD_KD:.2f}")
    print(f"PD_CENTER_THRESHOLD = {PD_CENTER_THRESHOLD:.3f}")
    print(f"PD_MIN_PULSE = {PD_MIN_PULSE:.3f}s")
    print(f"PD_MAX_PULSE = {PD_MAX_PULSE:.3f}s")
    print(f"PD_MOVEMENT_COOLDOWN = {PD_MOVEMENT_COOLDOWN:.3f}s")
    print("--------------------------------")

def print_help():
    """Print help information for terminal controls."""
    print("\n--- Terminal Controls ---")
    print("q          - Quit the program")
    print("h          - Show this help message")
    print("p          - Print current PD control settings")
    print("1/2        - Decrease/Increase PD_KP")
    print("3/4        - Decrease/Increase PD_KD")
    print("5/6        - Decrease/Increase PD_CENTER_THRESHOLD")
    print("7/8        - Decrease/Increase PD_MIN_PULSE")
    print("9/0        - Decrease/Increase PD_MAX_PULSE")
    print("-/+        - Decrease/Increase PD_MOVEMENT_COOLDOWN")
    print("--------------------------------")

def input_thread_function():
    """Thread function to handle terminal input."""
    global running, PD_KP, PD_KD, PD_CENTER_THRESHOLD, PD_MIN_PULSE, PD_MAX_PULSE, PD_MOVEMENT_COOLDOWN
    
    print_help()
    
    while running:
        # Check if there's input available (non-blocking)
        if select.select([sys.stdin], [], [], 0.1)[0]:
            key = sys.stdin.readline().strip()
            
            if key == 'q':
                print("Quitting program...")
                running = False
            elif key == 'h':
                print_help()
            elif key == 'p':
                print_current_settings()
            elif key == '1':
                PD_KP = max(0.1, PD_KP - 0.1)
                print(f"Decreased PD_KP to {PD_KP:.2f}")
            elif key == '2':
                PD_KP = min(2.0, PD_KP + 0.1)
                print(f"Increased PD_KP to {PD_KP:.2f}")
            elif key == '3':
                PD_KD = max(0.0, PD_KD - 0.05)
                print(f"Decreased PD_KD to {PD_KD:.2f}")
            elif key == '4':
                PD_KD = min(1.0, PD_KD + 0.05)
                print(f"Increased PD_KD to {PD_KD:.2f}")
            elif key == '5':
                PD_CENTER_THRESHOLD = max(0.01, PD_CENTER_THRESHOLD - 0.01)
                print(f"Decreased PD_CENTER_THRESHOLD to {PD_CENTER_THRESHOLD:.3f}")
            elif key == '6':
                PD_CENTER_THRESHOLD = min(0.2, PD_CENTER_THRESHOLD + 0.01)
                print(f"Increased PD_CENTER_THRESHOLD to {PD_CENTER_THRESHOLD:.3f}")
            elif key == '7':
                PD_MIN_PULSE = max(0.005, PD_MIN_PULSE - 0.005)
                print(f"Decreased PD_MIN_PULSE to {PD_MIN_PULSE:.3f}s")
            elif key == '8':
                PD_MIN_PULSE = min(PD_MAX_PULSE - 0.01, PD_MIN_PULSE + 0.005)
                print(f"Increased PD_MIN_PULSE to {PD_MIN_PULSE:.3f}s")
            elif key == '9':
                PD_MAX_PULSE = max(PD_MIN_PULSE + 0.01, PD_MAX_PULSE - 0.01)
                print(f"Decreased PD_MAX_PULSE to {PD_MAX_PULSE:.3f}s")
            elif key == '0':
                PD_MAX_PULSE = min(0.5, PD_MAX_PULSE + 0.01)
                print(f"Increased PD_MAX_PULSE to {PD_MAX_PULSE:.3f}s")
            elif key == '-':
                PD_MOVEMENT_COOLDOWN = max(0.01, PD_MOVEMENT_COOLDOWN - 0.05)
                print(f"Decreased PD_MOVEMENT_COOLDOWN to {PD_MOVEMENT_COOLDOWN:.3f}s")
            elif key == '=':
                PD_MOVEMENT_COOLDOWN = min(1.0, PD_MOVEMENT_COOLDOWN + 0.05)
                print(f"Increased PD_MOVEMENT_COOLDOWN to {PD_MOVEMENT_COOLDOWN:.3f}s")


# --- Core Functions ---

def setup_gpio():
    """Initialize GPIO pins for relays."""
    try:
        GPIO.setmode(GPIO.BCM) # Use Broadcom pin numbers
        GPIO.setwarnings(False)

        # Initialize all relay pins as outputs
        for pin in RELAY_PINS.values():
            GPIO.setup(pin, GPIO.OUT)

        # Set squirt relay to OFF state (LOW)
        print(f"Setting SQUIRT relay (pin {RELAY_PINS['squirt']}) to OFF state (LOW)...")
        GPIO.output(RELAY_PINS['squirt'], SQUIRT_RELAY_OFF_STATE)  # LOW

        # Set movement and unused relays to OFF state (HIGH for active-low)
        for name, pin in RELAY_PINS.items():
            if name not in ['squirt']:  # Skip the squirt relay (already set)
                off_state = GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW
                print(f"Setting {name} relay (pin {pin}) to OFF state ({'HIGH' if off_state == GPIO.HIGH else 'LOW'})...")
                GPIO.output(pin, off_state)

        print("GPIO Initialized.")
        print(f"Relay Pins: {RELAY_PINS}")
        print(f"Relays Active Low: {RELAY_ACTIVE_LOW}")
        print(f"Squirt Relay: ON=HIGH, OFF=LOW")
        print(f"Movement Relays: ON=LOW, OFF=HIGH (active-low)")

    except Exception as e:
        print(f"CRITICAL: Error initializing GPIO: {e}")
        print("Check GPIO connections and permissions.")
        raise # Propagate error to stop execution if GPIO fails

def setup_camera():
    """Setup PiCamera2 for capture."""
    print("Setting up Pi camera...")
    try:
        picam2 = Picamera2()
        
        # Configure the camera with optimized settings for detection
        camera_config = picam2.create_video_configuration(
            main={
                "size": (FRAME_WIDTH, FRAME_HEIGHT),
                "format": "XBGR8888"  # Use XBGR format for better handling with OpenCV
            },
            controls=CAMERA_SETTINGS
        )
        
        picam2.configure(camera_config)
        
        # Add noise reduction settings if supported
        try:
            picam2.set_controls({"NoiseReductionMode": 2})  # Enhanced noise reduction
        except Exception as nr_e:
            print(f"Note: Could not set noise reduction mode: {nr_e}")
        
        print(f"Pi camera configured: {FRAME_WIDTH}x{FRAME_HEIGHT} @ {TARGET_FPS} FPS")
        print("Applied Camera Settings:")
        for setting, value in CAMERA_SETTINGS.items():
            print(f"  {setting}: {value}")
        
        # Start the camera
        picam2.start()
        print("Camera startup initiated...")
        
        # Wait for camera to initialize
        time.sleep(2.0)  # Allow camera more time to stabilize
        
        # Verify camera is working by capturing a test frame
        try:
            test_frame = picam2.capture_array()
            print(f"Test frame captured: {test_frame.shape} (shape should match target resolution)")
            if test_frame.shape[0] != FRAME_HEIGHT or test_frame.shape[1] != FRAME_WIDTH:
                print(f"Warning: Test frame size {test_frame.shape[:2]} doesn't match target {(FRAME_HEIGHT, FRAME_WIDTH)}")
        except Exception as tf_e:
            print(f"Warning: Could not capture test frame: {tf_e}")
        
        print("Camera started successfully.")
        return picam2
        
    except Exception as e:
        print(f"CRITICAL: Error setting up Pi camera: {e}")
        print("Check camera connection, configuration, and libcamera support.")
        traceback.print_exc()
        raise

def setup_sound():
    """Initialize pygame mixer for sound effects if available."""
    if SOUND_ENABLED:
        try:
            pygame.mixer.init()
            print("Pygame mixer initialized for sound.")
        except Exception as e:
            print(f"Warning: Failed to initialize pygame mixer: {e}")
            return False
        return True
    return False

def load_model(model_name, zoo_path):
    """Load the specified Degirum model."""
    try:
        print(f"Loading Degirum model: {model_name} from {zoo_path}")

        # Create logs directory for HailoRT logs if it doesn't exist
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        os.chmod(log_dir, 0o777) # Ensure write permissions
        
        # Use absolute path for logs and set environment variable
        log_path = os.path.join(os.path.abspath(os.getcwd()), log_dir)
        os.environ["HAILORT_LOG_PATH"] = log_path
        print(f"HailoRT logs redirected to: {os.environ['HAILORT_LOG_PATH']}")

        # Make sure the log directory is writable
        try:
            test_log = os.path.join(log_path, "test_write.log")
            with open(test_log, 'w') as f:
                f.write("Test log write permission\n")
            os.remove(test_log)
            print("Successfully tested log directory write permissions")
        except Exception as log_e:
            print(f"Warning: Could not write to log directory: {log_e}")
            print("Try running with sudo or fix permissions on logs directory")

        if not os.path.exists(zoo_path):
            print(f"ERROR: Model zoo path not found: {zoo_path}")
            return None

        # Construct the full path to the model file (assuming .hef extension)
        # Degirum usually handles finding the .hef within the named folder
        model_path_check = os.path.join(zoo_path, model_name)
        if not os.path.exists(model_path_check):
             print(f"Warning: Model directory not found at {model_path_check}. Degirum will try to find it.")
             # List available models for easier debugging if needed
             try:
                 print("Available models/folders in zoo:")
                 for item in os.listdir(zoo_path):
                     print(f" - {item}")
             except Exception as list_e:
                 print(f"Could not list zoo contents: {list_e}")

        loaded_model = dg.load_model(
            model_name=model_name,
            inference_host_address=HAILO_INFERENCE_ADDRESS,
            zoo_url=zoo_path,
            output_confidence_threshold=DETECTION_THRESHOLD # Set confidence here
            # Removed overlay params for simplicity, handled by opencv drawing
        )
        print(f"Model '{model_name}' loaded successfully.")
        return loaded_model

    except Exception as e:
        print(f"CRITICAL: Failed to load model '{model_name}': {e}")
        traceback.print_exc()
        return None

def load_fallback_model(zoo_path):
    """Try loading common fallback models if the primary fails."""
    print("Attempting to load a fallback model...")
    fallback_models = [ # List potential models in your zoo
        "yolov5s_coco--640x640_quant_hailort_hailo8l_1",
        "yolov8n_coco--640x640_quant_hailort_hailo8l_1",
        "yolov8s_coco--640x640_quant_hailort_hailo8l_1",
        # Add other models present in your zoo_path
    ]
    
    # List available models in the zoo for reference
    try:
        print("Available models in zoo:")
        models_found = []
        for item in os.listdir(zoo_path):
            item_path = os.path.join(zoo_path, item)
            if os.path.isdir(item_path):
                # Check if directory contains *.hef files
                hef_files = [f for f in os.listdir(item_path) if f.endswith('.hef')]
                if hef_files:
                    models_found.append(item)
                    print(f" - {item} (contains {len(hef_files)} .hef files)")
        
        # Add any found models to our fallback list
        for model_name in models_found:
            if model_name not in fallback_models and model_name != HAILO_MODEL_NAME:
                fallback_models.append(model_name)
                print(f"Added {model_name} to fallback list")
    except Exception as e:
        print(f"Could not enumerate models in zoo: {e}")
        
    # Try loading models from our expanded list
    for model_name in fallback_models:
         if model_name == HAILO_MODEL_NAME: 
             continue # Skip the one that failed
         print(f"Trying fallback model: {model_name}")
         model = load_model(model_name, zoo_path)
         if model:
             print(f"Successfully loaded fallback model: {model_name}")
             return model
    
    print("ERROR: Could not load any fallback models.")
    print("Check the following:")
    print("1. Ensure Hailo drivers are installed and device is properly connected")
    print("2. Verify DeGirum model zoo path is correct")
    print("3. Check file permissions on the model zoo directory")
    print("4. Try running with sudo if it's a permissions issue")
    return None


def read_frame(camera_instance):
    """Capture and return a frame from the PiCamera2 instance."""
    try:
        # Capture frame as numpy array
        frame = camera_instance.capture_array()
        
        # Handle different color formats that Picamera2 might return
        if frame.shape[2] == 4:  # If it has 4 channels (XBGR or XRGB)
            # Convert XBGR to BGR by dropping the X channel
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif frame.shape[2] == 3:  # If it's RGB (3 channels)
            # Convert RGB to BGR for OpenCV processing and display
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Verify shape
        if frame.shape[0] != FRAME_HEIGHT or frame.shape[1] != FRAME_WIDTH:
            # Resize if somehow the shape is wrong (shouldn't happen with config)
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            if VERBOSE_OUTPUT: print("Warning: Resized frame in read_frame")

        return frame
    except Exception as e:
        print(f"Error capturing frame: {e}")
        # Return None on error, allowing the main loop to handle it
        return None

def activate_relay(pin, duration):
    """
    Activates a specific relay for a given duration (in seconds).
    Handles differences in ON/OFF states for squirt vs. other relays.
    """
    global last_action, last_action_time # Update global state for display

    relay_name = "UNKNOWN"
    for name, p in RELAY_PINS.items():
        if p == pin:
            relay_name = name.upper()
            break

    try:
        # Determine ON/OFF GPIO levels
        if pin == RELAY_PINS['squirt']:
            # Squirt relay: HIGH = ON, LOW = OFF
            on_state = SQUIRT_RELAY_ON_STATE  # HIGH
            off_state = SQUIRT_RELAY_OFF_STATE  # LOW
            action_prefix = "SQUIRT"
        else:
            # Standard active-low movement relays: LOW = ON, HIGH = OFF
            on_state = GPIO.LOW if RELAY_ACTIVE_LOW else GPIO.HIGH
            off_state = GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW
            action_prefix = f"MOVE {relay_name}"

        # Activate
        GPIO.output(pin, on_state)
        action_msg = f"{action_prefix} ON ({duration:.3f}s)"
        if VERBOSE_OUTPUT: print(f"  Relay {relay_name} (Pin {pin}) -> ON (State {'HIGH' if on_state == GPIO.HIGH else 'LOW'})")

        # Update global state immediately for drawing function
        current_time = time.time()
        last_action = action_msg
        last_action_time = current_time

        # Wait
        time.sleep(duration)

        # Deactivate
        GPIO.output(pin, off_state)
        if VERBOSE_OUTPUT: print(f"  Relay {relay_name} (Pin {pin}) -> OFF (State {'HIGH' if off_state == GPIO.HIGH else 'LOW'})")

    except Exception as e:
        print(f"Error activating relay {relay_name} (Pin {pin}): {e}")
        # Ensure relay is turned off in case of error during sleep
        try:
             GPIO.output(pin, off_state)
        except Exception as e_off:
             print(f"  Error ensuring relay off: {e_off}")


def process_detections(frame, model_results):
    """
    Processes raw detection results from the Degirum model.
    Returns a list of detection dictionaries:
    [{'bbox': (x1, y1, x2, y2), 'score': score, 'category_id': class_id, 'label': label}, ...]
    Handles potential variations in Degirum result object structure.
    """
    detections = []
    height, width = frame.shape[:2]

    try:
        # Print raw results type for debugging
        if VERBOSE_OUTPUT:
            print(f"Raw results type: {type(model_results)}")
            if hasattr(model_results, 'results'):
                print(f"Results has 'results' attribute with {len(model_results.results)} items")
                
        # --- Process different result structures ---
        if hasattr(model_results, 'results') and isinstance(model_results.results, list):
            # Standard DeGirum output format with results attribute containing list
            result_list = model_results.results
            if VERBOSE_OUTPUT: 
                print(f"Processing {len(result_list)} detections from results.results list")
                
                # Debug: Print structure of first result
                if result_list and isinstance(result_list[0], dict):
                    print(f"First result keys: {list(result_list[0].keys())}")
                    
            # Process individual results in the list
            for detection_result in result_list:
                # Format 1: Dictionary with 'detections' key
                if isinstance(detection_result, dict) and 'detections' in detection_result:
                    for det in detection_result['detections']:
                        if 'bbox' in det and ('score' in det or 'confidence' in det) and ('class_id' in det or 'category_id' in det):
                            bbox = det['bbox']
                            score = float(det.get('score', det.get('confidence', 0.0)))
                            class_id = int(det.get('class_id', det.get('category_id', -1)))
                            
                            if class_id in CLASSES_TO_DETECT and score >= DETECTION_THRESHOLD:
                                # Process valid detection
                                x1, y1, x2, y2 = map(int, bbox)
                                
                                # Validate coordinates
                                x1 = max(0, min(width - 1, x1))
                                y1 = max(0, min(height - 1, y1))
                                x2 = max(0, min(width - 1, x2))
                                y2 = max(0, min(height - 1, y2))
                                
                                # Skip invalid boxes
                                if x2 <= x1 or y2 <= y1:
                                    continue
                                    
                                label = COCO_CLASSES.get(class_id, f"ClassID {class_id}")
                                detections.append({
                                    'bbox': (x1, y1, x2, y2),
                                    'score': score,
                                    'category_id': class_id,
                                    'label': label
                                })
                
                # Format 2: Dictionary with 'data' key (YOLO-style output)
                elif isinstance(detection_result, dict) and 'data' in detection_result:
                    data = detection_result['data']
                    
                    if isinstance(data, np.ndarray):
                        # Determine the format based on shape
                        if len(data.shape) == 4:  # [batch, grid_h, grid_w, values]
                            data = data.reshape(-1, data.shape[-1])
                        elif len(data.shape) == 3:  # [grid_h, grid_w, values]
                            data = data.reshape(-1, data.shape[-1])
                            
                        # Format 2a: Each row is [x1, y1, x2, y2, score, class_id]
                        if data.shape[1] == 6:
                            for box_data in data:
                                x1, y1, x2, y2, score, class_id = box_data
                                
                                # Convert to proper types
                                class_id = int(class_id)
                                score = float(score)
                                
                                if class_id in CLASSES_TO_DETECT and score >= DETECTION_THRESHOLD:
                                    # Convert coordinates if normalized (0-1)
                                    if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
                                        x1 *= width
                                        y1 *= height
                                        x2 *= width
                                        y2 *= height
                                        
                                    # Convert to integers and validate
                                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                                    x1 = max(0, min(width - 1, x1))
                                    y1 = max(0, min(height - 1, y1))
                                    x2 = max(0, min(width - 1, x2))
                                    y2 = max(0, min(height - 1, y2))
                                    
                                    # Skip invalid boxes
                                    if x2 <= x1 or y2 <= y1:
                                        continue
                                        
                                    label = COCO_CLASSES.get(class_id, f"ClassID {class_id}")
                                    detections.append({
                                        'bbox': (x1, y1, x2, y2),
                                        'score': score,
                                        'category_id': class_id,
                                        'label': label
                                    })
                        
                        # Format 2b: Each row has objectness + class scores
                        elif data.shape[1] > 5:  # [x_center, y_center, width, height, objectness, class_scores...]
                            for box_data in data:
                                objectness = box_data[4]
                                
                                if objectness >= DETECTION_THRESHOLD:
                                    # Find max class score and its index
                                    class_scores = box_data[5:]
                                    class_id = np.argmax(class_scores)
                                    score = float(class_scores[class_id])
                                    
                                    if class_id in CLASSES_TO_DETECT and score >= DETECTION_THRESHOLD:
                                        # Convert center format to corner format
                                        x_center, y_center, width, height = box_data[0:4]
                                        
                                        # Convert normalized coordinates if needed
                                        if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1:
                                            x_center *= width
                                            y_center *= height
                                            box_width = width * width
                                            box_height = height * height
                                        else:
                                            box_width = width
                                            box_height = height
                                            
                                        # Calculate corners
                                        x1 = int(x_center - box_width / 2)
                                        y1 = int(y_center - box_height / 2)
                                        x2 = int(x_center + box_width / 2)
                                        y2 = int(y_center + box_height / 2)
                                        
                                        # Validate coordinates
                                        x1 = max(0, min(width - 1, x1))
                                        y1 = max(0, min(height - 1, y1))
                                        x2 = max(0, min(width - 1, x2))
                                        y2 = max(0, min(height - 1, y2))
                                        
                                        # Skip invalid boxes
                                        if x2 <= x1 or y2 <= y1:
                                            continue
                                            
                                        label = COCO_CLASSES.get(class_id, f"ClassID {class_id}")
                                        detections.append({
                                            'bbox': (x1, y1, x2, y2),
                                            'score': score,
                                            'category_id': class_id,
                                            'label': label
                                        })
        
        # For any other object structures that might need handling 
        else:
            if isinstance(model_results, list):
                # Direct list of detections
                result_list = model_results
                if VERBOSE_OUTPUT: print(f"Processing {len(result_list)} detections from direct results list")
            else:
                # Fallback or handle other structures if necessary
                result_list = []
                if VERBOSE_OUTPUT: 
                    print(f"Warning: Unexpected model result type: {type(model_results)}")
                    if hasattr(model_results, '__dict__'):
                        print(f"Result attributes: {list(model_results.__dict__.keys())}")

    except Exception as e:
        print(f"Error processing model results: {e}")
        import traceback
        traceback.print_exc()

    if detections:
        print(f"Found {len(detections)} valid detections")
        
    # Sort detections by confidence (highest first) - useful for focusing on one target
    detections.sort(key=lambda d: d['score'], reverse=True)

    return detections


def handle_pd_control(bbox, frame_width):
    """
    Handles movement based on target position using a simpler zone-based approach 
    with PD control for smoother movement.
    Activates squirt function independently.
    Returns the calculated error for visualization.
    """
    global previous_error, last_movement_time, last_action, last_action_time
    global last_squirt_activation, last_detection_time

    x1, y1, x2, y2 = map(int, bbox)
    center_x = (x1 + x2) / 2
    current_time = time.time()

    # Update last detection time (used for inactivity reset)
    last_detection_time = current_time

    # --- Squirt Logic (Independent of Centering) ---
    if current_time - last_squirt_activation >= RELAY_SQUIRT_COOLDOWN:
        print(f"Target detected! Activating squirt relay for {RELAY_SQUIRT_DURATION:.2f}s")
        # Play sound effect if enabled and available
        if SOUND_ENABLED and SOUND_FILES:
             try:
                 sound_file = random.choice(SOUND_FILES)
                 sound = pygame.mixer.Sound(sound_file)
                 sound.play()
             except Exception as sound_e:
                 print(f"Warning: Could not play sound {sound_file}: {sound_e}")
        # Activate squirt relay
        activate_relay(RELAY_PINS['squirt'], RELAY_SQUIRT_DURATION) # activate_relay updates last_action
        last_squirt_activation = current_time # Update squirt cooldown timer

    # --- Zone-Based Tracking with PD Smoothing ---
    # Calculate normalized error (-1.0 to 1.0)
    current_error = ((center_x / frame_width) - 0.5) * 2
    
    # Calculate the derivative component for smoother transitions
    error_derivative = current_error - previous_error
    previous_error = current_error
    
    # Define tracking zones
    # These simplify the control logic while still benefiting from PD smoothing
    zones = {
        'left': {'range': (-1.0, -PD_CENTER_THRESHOLD), 'relay': 'right', 'action': 'MOVE RIGHT'},
        'center': {'range': (-PD_CENTER_THRESHOLD, PD_CENTER_THRESHOLD), 'relay': None, 'action': 'CENTER'},
        'right': {'range': (PD_CENTER_THRESHOLD, 1.0), 'relay': 'left', 'action': 'MOVE LEFT'}
    }
    
    # Determine which zone the target is in
    current_zone = None
    for zone, config in zones.items():
        low, high = config['range']
        if low <= current_error <= high:
            current_zone = zone
            break
    
    if current_zone is None:
        # Fallback - should never happen
        current_zone = 'center'
    
    if VERBOSE_OUTPUT:
        print(f"Target in {current_zone} zone (error: {current_error:.3f}, derivative: {error_derivative:.3f})")
    
    # Check if we need to move
    if current_zone != 'center' and (current_time - last_movement_time >= PD_MOVEMENT_COOLDOWN):
        # Get relay info
        relay_name = zones[current_zone]['relay']
        action_desc = zones[current_zone]['action']
        
        if relay_name:
            # Calculate pulse duration using PD
            # Base it mainly on proportional component but dampen with derivative
            base_duration = PD_KP * abs(current_error)
            
            # The derivative component reduces pulse duration when error is decreasing
            # and increases it when error is increasing (same direction as error)
            derivative_adjustment = PD_KD * error_derivative * (1 if current_error > 0 else -1)
            
            # Calculate final pulse duration
            pulse_duration = base_duration + derivative_adjustment
            
            # Clamp to min/max values
            pulse_duration = max(PD_MIN_PULSE, min(pulse_duration, PD_MAX_PULSE))
            
            # Activate the relay
            print(f"PD {action_desc} (Error: {current_error:.3f}, Derivative: {error_derivative:.3f}, Duration: {pulse_duration:.3f}s)")
            activate_relay(RELAY_PINS[relay_name], pulse_duration)
            last_movement_time = current_time
            
    return current_error  # Return for visualization


def draw_frame_elements(frame, fps, detections, current_pd_error=None):
    """Draws FPS, detections, status text, and PD visualization onto the frame."""
    height, width = frame.shape[:2]
    display_frame = frame.copy() # Work on a copy

    # --- FPS Display ---
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(display_frame, fps_text, (width - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)

    # --- Active Relay Status ---
    if time.time() - last_action_time < 2.0:
        relay_status = f"Relay: {last_action}"
        cv2.putText(display_frame, relay_status, (width - 240, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 2)

    # --- PD Control Parameters Display ---
    pd_settings = f"KP: {PD_KP:.2f} KD: {PD_KD:.2f} TH: {PD_CENTER_THRESHOLD:.2f}"
    cv2.putText(display_frame, pd_settings, (10, height - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

    # --- Detections Display ---
    target_drawn = False
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        score = det['score']
        class_id = det['category_id']
        label = det['label']

        # Use different color for the primary target (first in sorted list)
        color = COLORS[0] if not target_drawn else COLORS[class_id % len(COLORS)]
        thickness = 3 if not target_drawn else 2

        # Draw bounding box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)

        # Draw label text with background
        text = f"{label}: {score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(display_frame, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
        cv2.putText(display_frame, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

        target_drawn = True # Mark that the primary target bbox is drawn

    # --- Status Text ---
    status_text = f"Last Action: {last_action}"
    # Display for 2 seconds after action
    if time.time() - last_action_time > 2.0:
         status_text = "Last Action: ---"

    cv2.putText(display_frame, status_text, (10, height - 10),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 2)

    # --- PD Control Visualization ---
    # Center line
    center_x_frame = width // 2
    cv2.line(display_frame, (center_x_frame, 0), (center_x_frame, height), COLOR_YELLOW, 1)

    if current_pd_error is not None:
         # Draw target position marker based on error
         marker_pos_x = int(center_x_frame + (width / 2) * current_pd_error)
         cv2.circle(display_frame, (marker_pos_x, height - 40), 8, COLOR_RED, -1)
         cv2.line(display_frame, (marker_pos_x, height-40), (marker_pos_x, height-20), COLOR_RED, 2)

         # Draw dead zone boundaries
         left_deadzone = int(center_x_frame - (width / 2) * PD_CENTER_THRESHOLD)
         right_deadzone = int(center_x_frame + (width / 2) * PD_CENTER_THRESHOLD)
         cv2.line(display_frame, (left_deadzone, height - 25), (left_deadzone, height - 15), COLOR_GREEN, 2)
         cv2.line(display_frame, (right_deadzone, height - 25), (right_deadzone, height - 15), COLOR_GREEN, 2)

         # Display error value
         error_text = f"PD Error: {current_pd_error:.3f}"
         cv2.putText(display_frame, error_text, (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
    else:
         # Indicate no target detected for PD
         no_target_text = "No Target for PD"
         cv2.putText(display_frame, no_target_text, (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

    # --- Add timestamp ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(display_frame, timestamp, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

    return display_frame


def cleanup():
    """Clean up resources before exiting."""
    print("\n--- Cleaning Up ---")
    global video_writer, camera, running

    # Signal input thread to exit
    running = False
    time.sleep(0.5)  # Short delay to allow thread to respond

    # Release video writer
    if video_writer is not None:
        try:
            video_writer.release()
            print("Video writer released.")
            video_writer = None
        except Exception as e:
            print(f"Error releasing video writer: {e}")

    # Stop camera
    if camera is not None:
        try:
            camera.stop()
            print("Camera stopped.")
            camera = None
        except Exception as e:
            print(f"Error stopping camera: {e}")

    # Turn off all relays with their correct OFF states
    try:
        print("Turning off all relays...")
        
        # Ensure GPIO mode is set (prevents errors if cleanup is called without setup)
        try:
            current_mode = GPIO.getmode()
            if current_mode is None:
                print("GPIO mode was not set, setting to BCM mode")
                GPIO.setmode(GPIO.BCM)
        except Exception as mode_e:
            print(f"Warning: Could not check/set GPIO mode: {mode_e}")
            print("Setting GPIO mode to BCM")
            GPIO.setmode(GPIO.BCM)
        
        # Ensure squirt relay is OFF (LOW)
        GPIO.output(RELAY_PINS['squirt'], SQUIRT_RELAY_OFF_STATE)
        print(f"Set squirt relay (pin {RELAY_PINS['squirt']}) to OFF state (LOW)")
        
        # Ensure all other relays are OFF (HIGH for active-low)
        for name, pin in RELAY_PINS.items():
            if name != 'squirt':
                off_state = GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW
                GPIO.output(pin, off_state)
                print(f"Set {name} relay (pin {pin}) to OFF state ({'HIGH' if off_state == GPIO.HIGH else 'LOW'})")
                
        time.sleep(0.2)  # Short pause to ensure state changes take effect
        
        # Now clean up GPIO resources
        GPIO.cleanup()
        print("GPIO cleaned up.")
    except Exception as e:
        print(f"Error during GPIO cleanup: {e}")

    # Quit pygame mixer
    if SOUND_ENABLED:
        try:
            pygame.mixer.quit()
            print("Pygame mixer quit.")
        except Exception as e:
            print(f"Error quitting pygame mixer: {e}")

    print("Cleanup complete.")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global running
    print("\nCtrl+C detected. Exiting gracefully...")
    running = False
    # Cleanup will be called in the finally block of main


# --- Utility Classes / Functions ---

class FPSCounter:
    """Simple class to calculate Frames Per Second."""
    def __init__(self):
        self._start = time.monotonic()
        self._count = 0
        self._fps = 0.0

    def update(self):
        self._count += 1

    def get_fps(self):
        now = time.monotonic()
        elapsed = now - self._start
        if elapsed >= 1.0:
            self._fps = self._count / elapsed
            self._start = now
            self._count = 0
        return self._fps

def init_video_writer():
    """Initialize video writer for recording."""
    if not RECORD_VIDEO and time.time() - program_start_time > VIDEO_START_RECORD_DURATION:
         # Don't start if video recording is disabled AND initial period has passed
         return None

    try:
        os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = os.path.join(VIDEO_OUTPUT_DIR, f"detection_{timestamp}.mp4")

        writer = cv2.VideoWriter(
            video_filename, VIDEO_CODEC, VIDEO_FPS, VIDEO_RESOLUTION
        )

        if not writer.isOpened():
            print(f"ERROR: Could not open video writer for {video_filename}. Check codec and permissions.")
            return None

        print(f"Video recording started: {video_filename}")
        return writer
    except Exception as e:
        print(f"Error initializing video writer: {e}")
        return None

def save_debug_frame(frame, base_filename="detection"):
     """Saves a frame for debugging, managing old files."""
     global frame_num_global # Use a global counter to ensure unique names over time

     # --- Cleanup old frames ---
     try:
         debug_files = sorted([f for f in os.listdir('.') if f.startswith(base_filename + '_') and f.endswith('.jpg')])
         while len(debug_files) >= MAX_SAVED_FRAMES:
             file_to_remove = debug_files.pop(0)
             os.remove(file_to_remove)
             if VERBOSE_OUTPUT: print(f"Removed old debug frame: {file_to_remove}")
     except Exception as e:
         print(f"Warning: Error cleaning up debug frames: {e}")

     # --- Save new frame ---
     try:
         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
         filename = f"{base_filename}_{timestamp}_{frame_num_global}.jpg"
         cv2.imwrite(filename, frame)
         if VERBOSE_OUTPUT: print(f"Saved debug frame: {filename}")
     except Exception as e:
         print(f"Error saving debug frame: {e}")

def debug_print(message, force=False):
    """Print a debug message only if DEBUG_MODE is enabled or force is True"""
    if DEBUG_MODE or force:
        print(f"[DEBUG] {message}")


# --- Test Functions ---

def test_relays():
    """Briefly activate each relay sequentially."""
    print("\n--- Testing Relays ---")
    setup_gpio() # Ensure GPIO is set up

    # Test squirt relay
    print("Testing SQUIRT relay...")
    activate_relay(RELAY_PINS['squirt'], 0.3)
    time.sleep(0.5)

    # Test left movement relay (should turn platform RIGHT)
    print("Testing LEFT relay (turn platform RIGHT)...")
    activate_relay(RELAY_PINS['right'], 0.3) # Remember right relay turns left
    time.sleep(0.5)

    # Test right movement relay (should turn platform LEFT)
    print("Testing RIGHT relay (turn platform LEFT)...")
    activate_relay(RELAY_PINS['left'], 0.3) # Remember left relay turns right
    time.sleep(0.5)

    print("Relay test complete.\n")

def test_model_on_sample(model_instance):
    """Run inference on a sample image if available."""
    sample_image_path = ensure_sample_image("sample_cat.jpg")
    if not sample_image_path:
        print("Could not get a sample image for testing, skipping test.")
        return

    print(f"\n--- Testing Model on Sample Image: {sample_image_path} ---")
    try:
        img = cv2.imread(sample_image_path)
        if img is None:
            print("Error: Failed to load sample image.")
            return

        # Preprocess image for model inference (consistent with main loop)
        preprocessed_img = preprocess_frame(img, MODEL_INPUT_SIZE)
        
        print(f"Sample image shape after preprocessing: {preprocessed_img.shape}")

        print("Running inference on sample...")
        # Use predict_batch as it's common for Degirum models
        results_generator = model_instance.predict_batch([preprocessed_img])
        results = next(results_generator) # Get the result for the single image

        print("Processing sample results...")
        detections = process_detections(preprocessed_img, results)

        print(f"Found {len(detections)} objects in sample.")
        
        # Print detection details for debugging
        for i, det in enumerate(detections):
            print(f"  Detection {i+1}: {det['label']} ({det['score']:.2f}) at {det['bbox']}")
            
        annotated_frame = draw_frame_elements(preprocessed_img, 0, detections) # Draw results

        save_filename = "sample_detection_output.jpg"
        cv2.imwrite(save_filename, annotated_frame)
        print(f"Saved annotated sample output to: {save_filename}")
        print("--- Sample Test Complete ---\n")

    except Exception as e:
        print(f"Error during sample image test: {e}")
        traceback.print_exc()

def test_degirum_setup():
     """Basic check for DeGirum and Hailo."""
     print("\n--- Checking DeGirum and Hailo Setup ---")
     try:
         import degirum
         print(f"DeGirum version: {degirum.__version__}")
     except ImportError:
         print("ERROR: DeGirum package not found. Please install it.")
         return False

     try:
         import subprocess
         # Create logs directory for HailoRT logs if it doesn't exist
         log_dir = "logs"
         os.makedirs(log_dir, exist_ok=True)
         os.chmod(log_dir, 0o777)  # Ensure write permissions
         
         # Use absolute path for logs and set environment variable
         log_path = os.path.join(os.path.abspath(os.getcwd()), log_dir)
         os.environ["HAILORT_LOG_PATH"] = log_path
         
         # Test log directory write access
         try:
             test_log = os.path.join(log_path, "test_write.log")
             with open(test_log, 'w') as f:
                 f.write("Test log write permission\n")
             os.remove(test_log)
             print("Successfully tested log directory write permissions")
         except Exception as log_e:
             print(f"Warning: Could not write to log directory: {log_e}")
             print("Try running with sudo or fix permissions on logs directory")
         
         # Try to run hailortcli device show command
         print("Running 'hailortcli device show' to check Hailo device...")
         try:
             result = subprocess.run(['hailortcli', 'device', 'show'],
                                    capture_output=True, text=True, timeout=10)
             
             print("Hailo device info (hailortcli):")
             print(result.stdout)
             
             if result.returncode != 0:
                 print(f"Warning: hailortcli command failed with code {result.returncode}")
                 print(f"Stderr: {result.stderr}")
                 if result.returncode == 106:
                     print("Error code 106 often indicates permission issues with log files or device access")
                     print("Try running with sudo or fix Hailo device permissions")
                 return True  # Continue even if command fails - the module itself might work
             
             print("Hailo device found and accessible")
         except FileNotFoundError:
             print("WARNING: 'hailortcli' command not found but DeGirum module is available")
             print("The HailoRT CLI tools may not be in PATH but the Python module may still work")
             return True  # Module availability is more important than CLI tools
         except Exception as cmd_e:
             print(f"Warning: Error running hailortcli command: {cmd_e}")
             print("Will try to continue with DeGirum module which might still work")
             return True
             
         print("--- Setup Check Complete ---\n")
         return True
     except Exception as e:
         print(f"Error checking Hailo runtime: {e}")
         print("Will attempt to continue with DeGirum module which might still work")
         return True

# --- Main Execution ---

def find_model_zoo_path():
    """
    Locate the Degirum model zoo path.
    Checks several common locations and returns the first valid path.
    """
    # Default path from configuration
    paths_to_check = [
        HAILO_ZOO_PATH,
        # Alternative potential locations
        "/home/pi/degirum_model_zoo",
        "/opt/degirum/model_zoo",
        "/home/pi5/degirum_model_zoo",  # Explicitly included
        "./degirum_model_zoo",
        os.path.expanduser("~/degirum_model_zoo")
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"Found model zoo at: {path}")
            try:
                # Check if it contains at least one model
                model_count = 0
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        model_count += 1
                
                if model_count > 0:
                    print(f"Model zoo contains {model_count} potential model directories")
                    return path
                else:
                    print(f"Warning: Path {path} exists but contains no model directories")
            except Exception as e:
                print(f"Error accessing {path}: {e}")
    
    print("WARNING: Could not find a valid model zoo path")
    print("Using default path despite issues, model loading will likely fail")
    return HAILO_ZOO_PATH

def main():
    global video_writer, camera, model, previous_error, last_detection_time
    global program_start_time, frame_num_global, running # Access global variables

    program_start_time = time.time() # Record script start time

    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    print("\n===============================================")
    print("=   Cat/Dog Detection System with PD Control  =")
    print("===============================================")
    print("Press 'h' for help on keyboard controls")

    # --- Initialization ---
    if not test_degirum_setup(): 
        print("WARNING: DeGirum setup check failed, but will try to continue")
        
    try:    
        test_relays() # Test relays early
    except Exception as e:
        print(f"WARNING: Relay test failed: {e}")
        print("This could be due to permissions. Try running with sudo.")
        
    setup_sound() # Initialize sound if available

    # Find the model zoo path
    model_zoo_path = find_model_zoo_path()
    
    # Try loading the model
    model = load_model(HAILO_MODEL_NAME, model_zoo_path)
    if model is None:
        print("Primary model failed to load, trying fallbacks...")
        model = load_fallback_model(model_zoo_path)
    if model is None:
        print("CRITICAL: Failed to load any model.")
        print("Will exit in 3 seconds...")
        time.sleep(3)
        sys.exit(1)

    # Print model information
    try:
        if hasattr(model, 'model_name'):
            print(f"Model name: {model.model_name}")
        if hasattr(model, 'input_shape'):
            print(f"Model input shape: {model.input_shape}")
        if hasattr(model, 'output_names'):
            print(f"Model output names: {model.output_names}")
        print(f"Detection threshold: {DETECTION_THRESHOLD}")
        print(f"Classes to detect: {CLASSES_TO_DETECT} ({[COCO_CLASSES.get(cls_id, f'Unknown {cls_id}') for cls_id in CLASSES_TO_DETECT]})")
    except Exception as e:
        print(f"Warning: Could not print model information: {e}")

    # Skip sample image test - focus on live camera detection
    
    try:
        camera = setup_camera()
    except Exception as e:
        print(f"CRITICAL: Camera setup failed: {e}")
        print("Cannot continue without camera. Will exit in 3 seconds...")
        time.sleep(3)
        sys.exit(1)

    # Initialize video writer (might start recording immediately based on settings)
    video_writer = init_video_writer()
    video_start_time = time.time() # Track current video file start time

    # Start input thread for keyboard commands
    input_thread = threading.Thread(target=input_thread_function)
    input_thread.daemon = True  # Thread will exit when main program exits
    input_thread.start()

    print("\n--- Starting Main Loop ---")
    fps_counter = FPSCounter()
    last_fps_print_time = time.time()
    frame_count = 0
    current_target_error = None # Store the error of the primary target for display

    while running:
        loop_start = time.monotonic()
        frame_num_global += 1 # Increment global frame counter

        # --- Read Frame ---
        frame = read_frame(camera)
        if frame is None:
            print("Warning: Failed to read frame, skipping iteration.")
            time.sleep(0.1) # Avoid busy-looping on error
            continue

        # --- Inference (Periodic) ---
        detections = []
        run_inference = (frame_count % FRAME_SKIP == 0)
        if run_inference:
            try:
                # Print basic camera frame info
                if VERBOSE_OUTPUT and frame_count % 30 == 0:  # Print every 30 frames
                    print(f"Camera frame: {frame.shape}, dtype: {frame.dtype}, min/max: {np.min(frame)}/{np.max(frame)}")
                
                # Preprocess frame for the model
                preprocessed_frame = preprocess_frame(frame, MODEL_INPUT_SIZE)
                
                # Log information about the frame
                if VERBOSE_OUTPUT and frame_count % 30 == 0:  # Print every 30 frames
                    print(f"Frame shape before inference: {preprocessed_frame.shape}")
                    print(f"Frame dtype: {preprocessed_frame.dtype}")
                    print(f"Frame min/max values: {np.min(preprocessed_frame)}/{np.max(preprocessed_frame)}")
                    if DEBUG_MODE:
                        # Save a debug frame to disk
                        debug_filename = f"debug_frame_{frame_count}.jpg"
                        cv2.imwrite(debug_filename, preprocessed_frame)
                        print(f"Saved debug frame to {debug_filename}")
                
                # Perform inference using the loaded Degirum model
                inference_start = time.time()
                
                # Create a batch with a single frame
                results_generator = model.predict_batch([preprocessed_frame])
                results = next(results_generator) # Get result for the single frame
                inference_time = time.time() - inference_start
                
                if VERBOSE_OUTPUT:
                    print(f"Inference completed in {inference_time:.4f}s")
                    print(f"Results type: {type(results)}")
                    if hasattr(results, 'results'):
                        print(f"Results contains {len(results.results)} items")
                        
                # Process results into a standard format
                detections = process_detections(preprocessed_frame, results)
                if VERBOSE_OUTPUT:
                    print(f"Found {len(detections)} objects after processing")
                    for i, det in enumerate(detections):
                        print(f"  Detection {i+1}: {det['label']} ({det['score']:.2f}) at {det['bbox']}")

            except Exception as e:
                print(f"Error during inference or processing: {e}")
                traceback.print_exc()  # Print the full stack trace for debugging
                # Proceed with empty detections list on error
                detections = []

        # --- Action & Control ---
        current_target_error = None # Reset for this frame
        if detections:
            # Target the highest confidence detection
            target_detection = detections[0]

            # Check if it's a class we want to control/squirt
            if target_detection['category_id'] in CLASSES_TO_DETECT:
                # Call PD control handler - it handles movement and squirts
                current_target_error = handle_pd_control(
                    target_detection['bbox'],
                    FRAME_WIDTH
                )
        else:
             # No detections this frame
             if VERBOSE_OUTPUT: print("No detections this frame.")
             # Reset previous error if no detection for a while
             if time.time() - last_detection_time > POSITION_RESET_TIME:
                  if previous_error != 0.0:
                       print(f"Resetting PD 'previous_error' due to inactivity ({POSITION_RESET_TIME}s).")
                       previous_error = 0.0

        # --- Update FPS ---
        fps_counter.update()
        current_fps = fps_counter.get_fps()
        if time.time() - last_fps_print_time >= 5.0: # Print FPS every 5 seconds
             print(f"FPS: {current_fps:.1f}")
             last_fps_print_time = time.time()

        # --- Draw Visualization ---
        display_frame = draw_frame_elements(frame, current_fps, detections, current_target_error)

        # --- Video Recording ---
        # Always record for the first VIDEO_START_RECORD_DURATION seconds
        elapsed_program_time = time.time() - program_start_time
        should_be_recording = RECORD_VIDEO or (elapsed_program_time < VIDEO_START_RECORD_DURATION)

        if should_be_recording:
             if video_writer is None: # Start writer if needed and not already running
                 video_writer = init_video_writer()
                 if video_writer: video_start_time = time.time()

             if video_writer is not None:
                 try:
                     video_writer.write(display_frame)
                     # Check for max video length
                     if time.time() - video_start_time >= VIDEO_MAX_LENGTH:
                         print(f"Video segment reached max length ({VIDEO_MAX_LENGTH}s). Creating new file.")
                         video_writer.release()
                         video_writer = init_video_writer()
                         if video_writer: video_start_time = time.time()

                 except Exception as e:
                     print(f"Error writing video frame: {e}")
                     # Attempt to close and restart writer
                     try: video_writer.release()
                     except: pass
                     video_writer = None # Stop trying to write to broken writer

        elif video_writer is not None: # If recording should stop (initial duration ended & RECORD_VIDEO=False)
             print("Stopping initial video recording period.")
             video_writer.release()
             video_writer = None

        # --- Optional: Save Debug Frames ---
        if SAVE_DEBUG_FRAMES and detections and (frame_num_global % DEBUG_FRAME_INTERVAL == 0):
             save_debug_frame(display_frame)

        # --- Frame Rate Control ---
        loop_end = time.monotonic()
        loop_duration = loop_end - loop_start
        sleep_time = (1.0 / TARGET_FPS) - loop_duration
        if sleep_time > 0:
            time.sleep(sleep_time)

        frame_count += 1

    # End of main loop
    print("Main loop exited.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n--- UNHANDLED EXCEPTION IN MAIN ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        traceback.print_exc()
        print("Attempting cleanup...")
    finally:
        # Ensure cleanup runs even if main crashes
        cleanup()
        print("Program terminated.")

def preprocess_frame(frame, target_size=(640, 640)):
    """
    Preprocess a frame for model inference by resizing and normalizing.
    
    Args:
        frame: Input BGR frame from camera
        target_size: Target size (width, height) for the model input
        
    Returns:
        Preprocessed frame ready for model inference
    """
    # Resize frame to match model input size
    if frame.shape[0] != target_size[1] or frame.shape[1] != target_size[0]:
        resized = cv2.resize(frame, target_size)
        if VERBOSE_OUTPUT:
            print(f"Resized frame from {frame.shape[1]}x{frame.shape[0]} to {target_size[0]}x{target_size[1]}")
    else:
        resized = frame
    
    # Make sure we're in the expected color format (BGR for OpenCV)
    # The model might expect RGB, but the DeGirum library likely handles this conversion
    
    # Create a copy to avoid any issues with buffer overwriting
    processed = resized.copy()
    
    # Print shape for debugging
    if VERBOSE_OUTPUT:
        print(f"Preprocessed frame shape: {processed.shape}")
    
    return processed

def ensure_sample_image(filename="sample_cat.jpg"):
    """Ensure a sample image exists for testing, create one if it doesn't."""
    if os.path.exists(filename):
        print(f"Sample image exists: {filename}")
        return filename
    
    print(f"Sample image not found: {filename}. Will create a basic test image.")
    
    # Create a basic test image
    try:
        # Create a black image with a colored rectangle
        img = np.zeros((MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0], 3), dtype=np.uint8)
        
        # Draw a cat-like shape (simplified)
        cv2.rectangle(img, (200, 200), (400, 400), (0, 0, 255), -1)  # Red rectangle
        cv2.circle(img, (250, 250), 30, (255, 0, 0), -1)  # Blue circle (left eye)
        cv2.circle(img, (350, 250), 30, (255, 0, 0), -1)  # Blue circle (right eye)
        cv2.ellipse(img, (300, 350), (50, 20), 0, 0, 180, (255, 255, 255), -1)  # White ellipse (mouth)
        
        # Add text label
        cv2.putText(img, "Test Cat", (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save the image
        cv2.imwrite(filename, img)
        print(f"Created test image: {filename}")
        return filename
    except Exception as e:
        print(f"Error creating sample image: {e}")
        return None