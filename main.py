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
    'center': 5,    # Center relay (triggers on any detection)
    'left': 6,      # Left relay (triggers for left-side detections)
    'right': 13,    # Right relay (triggers for right-side detections)
    'unused': 15    # Unused relay
}

# Model Setup
inference_host_address = "@local"
zoo_url = "/home/pi5/degirum_model_zoo"
token = ""
model_name = "yolov8n_cats"  # Your custom YOLOv8n model

# Configuration
DETECTION_THRESHOLD = 0.25  # Increased from 0.1 to 0.25 for better quality detections
MODEL_INPUT_SIZE = (640, 640)  # YOLOv8n input size
CENTER_THRESHOLD = 0.1  # Threshold for determining if object is left/right of center
RELAY_CENTER_DURATION = 0.2  # Duration to activate center relay
RELAY_CENTER_COOLDOWN = 1.0  # Cooldown period for center relay
INFERENCE_INTERVAL = 0.2  # Run inference every 200ms to reduce load
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 15  # Reduced from 25 to 15 for better performance
DEBUG_MODE = False  # Turn off debug mode to reduce console output
VERBOSE_OUTPUT = False  # Controls detailed terminal output
SAVE_EVERY_FRAME = False  # Only save frames with detections to reduce disk I/O

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

# Global variables for relay control
last_center_activation = 0
last_action = None
last_action_time = 0
last_valid_overlay = None  # Stores the last valid image overlay from the model

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
        print("Setting up Pi camera...")
        try:
            from picamera2 import Picamera2
            
            # Create and configure Picamera2
            picam2 = Picamera2()
            
            # Configure camera with preview configuration
            camera_config = picam2.create_preview_configuration(
                main={"format": "RGB888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
            )
            
            # Apply the configuration
            picam2.configure(camera_config)
            
            # Start the camera
            picam2.start()
            
            print(f"Pi camera initialized with resolution {FRAME_WIDTH}x{FRAME_HEIGHT}")
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
            for pin in RELAY_PINS.values():
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)  # Initialize all relays to OFF
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
            # Convert from RGB to BGR for OpenCV
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
            # Check if there are results in the standard results list
            if hasattr(results, 'results') and results.results:
                if VERBOSE_OUTPUT:
                    print(f"Processing {len(results.results)} detections")
                
                # Process each detection
                for detection in results.results:
                    try:
                        # Method 1: Direct access if detection is a dictionary
                        if isinstance(detection, dict):
                            if 'bbox' in detection:
                                bbox = detection['bbox']
                                x1, y1, x2, y2 = map(int, bbox)
                                score = float(detection.get('score', 0.0))
                                
                                # Get the class ID depending on which model we're using
                                if USE_COCO_MODEL:
                                    class_id = int(detection.get('category_id', 0))
                                    # Only process cat class (15) or other specified classes
                                    if class_id not in CLASSES_TO_DETECT:
                                        continue
                                else:
                                    class_id = int(detection.get('category_id', 0))
                                
                                if VERBOSE_OUTPUT:
                                    print(f"Detection: bbox={x1},{y1},{x2},{y2}, score={score:.2f}")
                                detections.append((x1, y1, x2, y2, score, class_id))
                                continue
                        
                        # Method 2: Access attributes if they exist
                        if hasattr(detection, 'bbox'):
                            # If bbox is an object
                            if hasattr(detection.bbox, 'x1'):
                                x1 = int(detection.bbox.x1)
                                y1 = int(detection.bbox.y1)
                                x2 = int(detection.bbox.x2)
                                y2 = int(detection.bbox.y2)
                            # If bbox is a list/tuple/array
                            elif hasattr(detection.bbox, '__getitem__'):
                                x1 = int(detection.bbox[0])
                                y1 = int(detection.bbox[1])
                                x2 = int(detection.bbox[2])
                                y2 = int(detection.bbox[3])
                            else:
                                continue
                                
                            # Get score and class ID
                            score = getattr(detection, 'score', getattr(detection, 'confidence', 0.3))
                            class_id = getattr(detection, 'class_id', getattr(detection, 'category_id', 15))
                                
                            # For COCO model, only process specified classes
                            if USE_COCO_MODEL and class_id not in CLASSES_TO_DETECT:
                                class_id = 15  # Treat detections as cats if using COCO model
                                
                            detections.append((x1, y1, x2, y2, score, class_id))
                    except Exception as e:
                        if VERBOSE_OUTPUT:
                            print(f"Error processing detection: {e}")
            
            # If we have no detections but have an image_overlay, extract detections from it
            if len(detections) == 0 and hasattr(results, 'image_overlay') and results.image_overlay is not None:
                # Create a copy of the frame for using OpenCV contour detection
                overlay = results.image_overlay

                # Check if the overlay has proper dimensions and looks like a valid image
                if isinstance(overlay, np.ndarray) and overlay.shape == frame.shape:                    
                    # Analyze difference between original frame and overlay
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    overlay_gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
                    
                    # Calculate absolute difference
                    diff = cv2.absdiff(frame_gray, overlay_gray)
                    
                    # Threshold to find areas of difference (where boxes might be)
                    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                    
                    # Find contours in the difference image
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Process contours to find bounding boxes
                    for contour in contours:
                        # Filter out small contours that might be noise
                        area = cv2.contourArea(contour)
                        if area > 500:  # Minimum area threshold
                            x, y, w, h = cv2.boundingRect(contour)
                            
                            # Filter by aspect ratio to avoid UI elements
                            aspect_ratio = float(w) / h if h > 0 else 0
                            if 0.25 < aspect_ratio < 4.0 and w > 30 and h > 30:
                                # Create a detection entry
                                x1, y1, x2, y2 = x, y, x+w, y+h
                                score = 0.5  # Assumed score
                                class_id = 15  # Cat class ID
                                
                                detections.append((x1, y1, x2, y2, score, class_id))
                
                # Just use the whole image as a valid overlay
                if len(detections) == 0:
                    # Store the overlay for display
                    global last_valid_overlay
                    last_valid_overlay = overlay
        
        except Exception as e:
            if VERBOSE_OUTPUT:
                print(f"Error processing detection results: {e}")
    
    if VERBOSE_OUTPUT:
        print(f"Returning {len(detections)} processed detections")
    return detections

def activate_relay(pin, duration=0.1):
    """Activate a relay for the specified duration"""
    global last_action, last_action_time
    
    current_time = time.time()
    if not DEV_MODE:
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(pin, GPIO.LOW)
    else:
        # Simulate relay activation with messages and sound
        if pin == RELAY_PINS['center']:
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
    global last_center_activation
    
    x1, y1, x2, y2 = map(int, bbox)
    center_x = (x1 + x2) / 2
    relative_position = (center_x / frame_width) - 0.5  # -0.5 to 0.5 range
    
    current_time = time.time()
    
    # Check if we can activate the center relay (cooldown period)
    if current_time - last_center_activation >= RELAY_CENTER_COOLDOWN:
        activate_relay(RELAY_PINS['center'], RELAY_CENTER_DURATION)
        last_center_activation = current_time
    
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
    """Load the YOLOv8 model for detection."""
    try:
        if USE_COCO_MODEL:
            print("Loading YOLOv8n COCO model for Hailo accelerator...")
            model_to_load = "yolov8n_coco--640x640_quant_hailort_hailo8l_1"  # Use specific .hef model
            zoo_path = "/home/pi5/degirum_model_zoo"  # Use local path where the model exists
        else:
            print("Loading custom cat model for Hailo accelerator...")
            model_to_load = model_name
            zoo_path = zoo_url
        
        import degirum as dg
        
        # Check if model zoo path exists
        if not os.path.exists(zoo_path):
            print(f"ERROR: Model zoo path not found: {zoo_path}")
            print("Please check that the path is correct and the directory exists.")
            return None
            
        print(f"Model zoo path verified: {zoo_path}")
        print(f"Attempting to load model: {model_to_load}")
        
        # Check if the specific model path exists
        specific_model_path = os.path.join(zoo_path, model_to_load)
        if os.path.exists(specific_model_path):
            print(f"Found model at: {specific_model_path}")
        else:
            print(f"WARNING: Specific model path not found: {specific_model_path}")
        
        # List available models if possible
        try:
            available_models = os.listdir(zoo_path)
            print(f"Available files in model zoo directory: {available_models}")
        except Exception as e:
            print(f"Warning: Could not list contents of model zoo directory: {e}")
        
        # Load the model with local inference
        model = dg.load_model(
            model_name=model_to_load,
            inference_host_address=inference_host_address,  # Use @local for local inference
            zoo_url=zoo_path,  # Path to model zoo
            output_confidence_threshold=DETECTION_THRESHOLD,  # Use threshold from constants
            overlay_font_scale=2.5,  # Font scale for overlay
            overlay_show_probabilities=True  # Show confidence scores
        )
        
        print(f"Model loaded successfully with confidence threshold: {DETECTION_THRESHOLD}")
        
        # Try to access more model info
        if hasattr(model, 'info'):
            print(f"Model info: {model.info}")
        if hasattr(model, 'config'):
            print(f"Model config: {model.config}")
            
        # Print available model methods
        print(f"Model methods: {[m for m in dir(model) if not m.startswith('_') and callable(getattr(model, m))]}")
        
        return model
        
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        print("Detailed error traceback:")
        import traceback
        traceback.print_exc()
        print("\nCommon model loading issues:")
        print("1. Model zoo path incorrect or missing")
        print("2. Model file not found in the model zoo directory")
        print("3. DeGirum package not installed correctly")
        print("4. Hailo accelerator not connected or detected")
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
    
    if len(detections) > 0:
        if VERBOSE_OUTPUT:
            print(f"Detected {len(detections)} objects")
        
        # Process each detection
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
            
            # Calculate relative position for GPIO control
            object_center_x = (x1 + x2) / 2
            relative_position = (object_center_x / frame_width) - 0.5
            
            # Control GPIO based on position
            if not DEV_MODE:
                if relative_position < -CENTER_THRESHOLD:
                    # Object is on the left side
                    set_relay(RELAY_PIN_LEFT, True)
                    set_relay(RELAY_PIN_RIGHT, False)
                elif relative_position > CENTER_THRESHOLD:
                    # Object is on the right side
                    set_relay(RELAY_PIN_LEFT, False)
                    set_relay(RELAY_PIN_RIGHT, True)
                else:
                    # Object is centered
                    set_relay(RELAY_PIN_LEFT, False)
                    set_relay(RELAY_PIN_RIGHT, False)
            
            # Draw directional indicators
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
    else:
        # Turn off relays if no detections
        if not DEV_MODE:
            set_relay(RELAY_PIN_LEFT, False)
            set_relay(RELAY_PIN_RIGHT, False)
    
    # Add timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(annotated_frame, timestamp, (width - 250, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
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
                
                # Convert from RGB to BGR for OpenCV
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
        GPIO.setup(RELAY_PIN_LEFT, GPIO.OUT)
        GPIO.setup(RELAY_PIN_RIGHT, GPIO.OUT)
        
        # Initialize both relays to off
        GPIO.output(RELAY_PIN_LEFT, GPIO.LOW)
        GPIO.output(RELAY_PIN_RIGHT, GPIO.LOW)
        
        print(f"GPIO initialized: left relay on pin {RELAY_PIN_LEFT}, right relay on pin {RELAY_PIN_RIGHT}")
    except ImportError:
        print("WARNING: RPi.GPIO module not available, GPIO control disabled")
    except Exception as e:
        print(f"Error initializing GPIO: {e}")
        
def set_relay(pin, state):
    """Set relay state (True = ON, False = OFF)"""
    try:
        import RPi.GPIO as GPIO
        GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)
        if DEBUG_MODE:
            print(f"Relay {pin} set to {'ON' if state else 'OFF'}")
    except (ImportError, NameError):
        # If we can't import GPIO, just print what we would do
        if DEBUG_MODE:
            print(f"Would set relay {pin} to {'ON' if state else 'OFF'}")
    except Exception as e:
        print(f"Error setting relay {pin}: {e}")

def cleanup():
    """Clean up resources before exiting"""
    print("Cleaning up resources...")
    
    # Clean up GPIO
    try:
        if not DEV_MODE:
            import RPi.GPIO as GPIO
            GPIO.cleanup()
            print("GPIO pins cleaned up")
    except:
        pass
    
    # Release camera if it exists
    try:
        if 'camera' in globals() and camera is not None:
            camera.release()
            print("Camera released")
    except:
        pass

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
        
        # Test model on a sample image if it exists
        if os.path.exists("sample_cat.jpg") and not DEV_MODE:
            test_model_on_sample(model)
        
        # Initialize GPIO if not in dev mode
        if not DEV_MODE:
            init_gpio()
        
        # Print configuration
        print_config()
        
        # Initialize variables for smoothing detections
        last_valid_detections = []
        no_detection_frames = 0
        max_no_detection_frames = 5  # Keep using last detection for more frames
        
        # Track performance metrics
        fps_counter = FPSCounter()
        frame_count = 0
        last_fps_print = time.time()
        
        # Main loop
        while True:
            frame_start_time = time.time()
            
            # Read frame from camera
            frame = read_frame(camera)
            
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Only run inference every other frame to improve frame rate
            frame_count += 1
            should_run_inference = (frame_count % 2 == 0)
            
            if should_run_inference:
                # Perform inference
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
                        if VERBOSE_OUTPUT:
                            print(f"Using last valid detection (frame {no_detection_frames}/{max_no_detection_frames})")
                        detections = last_valid_detections
                else:
                    # We have valid detections, reset counter and save for future use
                    no_detection_frames = 0
                    last_valid_detections = detections.copy()
            else:
                # Use previous detections on frames where we skip inference
                detections = last_valid_detections
            
            # Get FPS
            fps = fps_counter.get_fps()
            
            # Process actions based on detections (output to relays)
            processed_frame = process_actions(frame, detections, fps)
            
            # Only save images if we have detections or if save every frame is enabled
            has_detections = len(detections) > 0
            if has_detections or SAVE_EVERY_FRAME:
                timestamp = int(time.time())
                # Save processed frame every 10th frame or on detection
                if has_detections or (frame_count % 10 == 0):
                    cv2.imwrite(f"latest_frame_{timestamp}.jpg", processed_frame)
            
            # Print FPS periodically
            current_time = time.time()
            if current_time - last_fps_print >= 5.0:  # Print every 5 seconds
                print(f"Current FPS: {fps:.1f}")
                last_fps_print = current_time
            
            # Calculate time spent on this frame and sleep if needed
            frame_time = time.time() - frame_start_time
            sleep_time = max(0, (1.0 / FPS) - frame_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting...")
        
    except Exception as e:
        print(f"Error in main loop: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up resources
        cleanup()
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