#!/usr/bin/env python3
"""
Test script for calibrating tracking relay activation times.
This script tests different relay activation durations to find the optimal time
for centering detected objects.
"""

import os
import cv2
import time
import numpy as np
import datetime
import argparse
import signal
import sys

# Import Pi-specific modules on Raspberry Pi
try:
    from picamera2 import Picamera2
    import RPi.GPIO as GPIO
    DEV_MODE = False
    print("Running in Raspberry Pi mode")
except ImportError:
    DEV_MODE = True
    print("Running in development mode (no GPIO)")

# Import DeGirum for Hailo AI Kit
try:
    import degirum as dg
    print("Successfully imported DeGirum module")
    DEGIRUM_AVAILABLE = True
except ImportError as e:
    print(f"Failed to import DeGirum: {e}")
    print("Make sure DeGirum is installed:")
    print("  pip install degirum")
    print("  or follow installation instructions at https://docs.degirum.com/")
    DEGIRUM_AVAILABLE = False
    print("DeGirum not available, detection will not work")

# Configuration
VIDEO_OUTPUT_DIR = "tracking_tests"
VIDEO_CODEC = cv2.VideoWriter_fourcc(*'mp4v')
VIDEO_FPS = 30
VIDEO_RESOLUTION = (640, 640)
FRAME_WIDTH, FRAME_HEIGHT = VIDEO_RESOLUTION

# Detection settings
DETECTION_THRESHOLD = 0.30
CENTER_THRESHOLD = 0.1
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear'
}
CLASSES_TO_DETECT = [15, 16]  # Cat and dog class IDs in COCO dataset

# GPIO Pin Setup
RELAY_PINS = {
    'squirt': 16,    # Squirt relay (triggers water gun)
    'left': 6,      # Left relay (triggers for left-side movement)
    'right': 13,    # Right relay (triggers for right-side movement)
    'unused': 15    # Unused relay
}

# Important relay configuration
RELAY_ACTIVE_LOW = True
RELAY_NORMALLY_CLOSED = False
SQUIRT_RELAY_ON_STATE = GPIO.HIGH if not DEV_MODE else 1
SQUIRT_RELAY_OFF_STATE = GPIO.LOW if not DEV_MODE else 0
RELAY_SQUIRT_DURATION = 0.2  # Duration to activate squirt relay in seconds

# Test duration settings
TEST_DURATION = 30  # Duration for each test phase in seconds (increased from 15s)
NUM_TEST_PHASES = 4  # Number of test phases

# Initial activation duration and increment
BASE_DURATION = 0.2  # seconds - starting with 0.2s
INCREMENT = 0.1      # seconds per phase - test 0.2s, 0.3s, 0.4s, 0.5s

# Camera settings optimized for better detection
CAMERA_SETTINGS = {
    "AeEnable": True,           # Auto exposure
    "AwbEnable": True,          # Auto white balance
    "AeExposureMode": 0,        # Normal exposure mode
    "AeMeteringMode": 0,        # Center-weighted metering
    "ExposureTime": 0,          # Let auto-exposure handle it
    "AnalogueGain": 1.0,        # Reduced gain to prevent washout
    "Brightness": 0.0,          # Reduced brightness to prevent washout
    "Contrast": 1.3,            # Increased contrast for better definition
    "Saturation": 1.1,          # Slightly increased saturation for better color
    "FrameRate": VIDEO_FPS,     # Set desired frame rate
    "AeConstraintMode": 0,      # Normal constraint mode
    "AwbMode": 1,               # Auto white balance mode (1 is typically auto)
    "ExposureValue": 0.0        # Reduced EV compensation to prevent overexposure
}

# Colors for visualization
COLORS = [
    (0, 255, 0),     # Green
    (255, 0, 0),     # Blue
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
]

# Global tracking variables - add these near the top with other globals
POSITION_RESET_TIME = 60.0  # Reset to center after 1 minute without detections
LAST_DETECTION_TIME = time.time()  # Time of last detection

# New zone-based tracking system
TRACKING_ZONES = {
    'far_left': {'range': (-1.0, -0.6), 'relay': 'right', 'duration': 0.20},
    'left': {'range': (-0.6, -0.2), 'relay': 'right', 'duration': 0.10},
    'center': {'range': (-0.2, 0.2), 'relay': None, 'duration': 0},
    'right': {'range': (0.2, 0.6), 'relay': 'left', 'duration': 0.10},
    'far_right': {'range': (0.6, 1.0), 'relay': 'left', 'duration': 0.20}
}
CURRENT_ZONE = 'center'  # Current tracking zone
MOVEMENT_COOLDOWN = 0.5  # Minimum time between movements (seconds)
LAST_MOVEMENT_TIME = 0  # Time of last movement

def resize_image_letterbox(image, target_shape=(640, 640)):
    """
    Resizes an image with letterboxing to preserve aspect ratio.
    """
    ih, iw = image.shape[:2]
    h, w = target_shape
    scale = min(w/iw, h/ih)
    
    # Resize image preserving aspect ratio
    nw, nh = int(iw * scale), int(ih * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    # Create empty target image
    new_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Compute letterbox margins
    top = (h - nh) // 2
    left = (w - nw) // 2
    
    # Copy resized image to center of target image
    new_image[top:top+nh, left:left+nw, :] = image
    
    return new_image, scale, top, left

def setup_camera():
    """Setup camera for capture"""
    if DEV_MODE:
        print("Setting up webcam for development mode...")
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
            print(f"Setting SQUIRT relay (pin {RELAY_PINS['squirt']}) to OFF state...")
            GPIO.output(RELAY_PINS['squirt'], SQUIRT_RELAY_OFF_STATE)
            
            # Initialize other relays to OFF state (HIGH for active-low relays)
            for name, pin in RELAY_PINS.items():
                if name != 'squirt':
                    print(f"Setting {name} relay (pin {pin}) to OFF state...")
                    GPIO.output(pin, GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW)
            
            print("Successfully initialized GPIO pins")
        except Exception as e:
            print(f"Failed to setup GPIO: {e}")
            raise

def load_model():
    """Load detection model - using same approach as main.py"""
    if not DEGIRUM_AVAILABLE:
        print("DeGirum not available, using OpenCV DNN instead")
        return None
        
    try:
        print("Loading DeGirum model...")
        # Use the same model name as in main.py
        model_name = "yolo11s_silu_coco--640x640_quant_hailort_hailo8l_1"
        
        # Try to load from local zoo first
        zoo_path = "/home/pi5/degirum_model_zoo"
        if os.path.exists(zoo_path):
            print(f"Loading from local model zoo: {zoo_path}")
            try:
                # First try with full path
                model = dg.load_model(
                    model_name=model_name,
                    inference_host_address="@local",
                    zoo_url=zoo_path
                )
                print("Successfully loaded model with full path")
            except Exception as e:
                print(f"Error loading model with full path: {e}")
                # Try with public model if local fails
                print("Falling back to yolov8n_coco from public model")
                model = dg.load_model(
                    model_name="yolov8n_coco",
                    inference_host_address="@local",
                    zoo_url="degirum/public"
                )
                print("Successfully loaded public yolov8n_coco model")
        else:
            # Fall back to public model
            print("Local model zoo not found, using yolov8n_coco from public model")
            model = dg.load_model(
                model_name="yolov8n_coco",
                inference_host_address="@local",
                zoo_url="degirum/public"
            )
            print("Successfully loaded public yolov8n_coco model")
            
        # Print model info for debugging
        print(f"Model loaded successfully: {model}")
        print(f"Model class: {model.__class__.__name__}")
        
        # List model attributes for debugging
        print("Model attributes:")
        for attr in dir(model):
            if not attr.startswith('_'):  # Skip private attributes
                try:
                    value = getattr(model, attr)
                    if not callable(value):  # Skip methods
                        print(f"  {attr}: {value}")
                except Exception as e:
                    print(f"  {attr}: Error accessing - {e}")
        
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        print(f"Detailed error info: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nCheck that the Hailo AI Kit is properly connected and initialized.")
        print("Make sure the DeGirum service is running:")
        print("  sudo systemctl status degirum")
        print("If not active, start it with:")
        print("  sudo systemctl start degirum")
        print("If issues persist, try reinstalling the DeGirum package.")
        return None

def get_frame(camera):
    """Get frame from camera based on mode"""
    if DEV_MODE:
        ret, frame = camera.read()
        if not ret:
            raise Exception("Failed to grab frame from webcam")
        # Resize frame to match model input size
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
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
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            return frame
        except Exception as e:
            print(f"Failed to capture frame from Pi camera: {e}")
            raise

def process_detections(results):
    """Process detection results and return detections list"""
    detections = []
    
    if results is None:
        return []
    
    try:
        # Print raw results for debugging
        print(f"Raw results type: {type(results)}")
        if hasattr(results, 'results'):
            print(f"Results has 'results' attribute with {len(results.results)} items")
            
        # Check if the format is a direct detection array (common in DeGirum)
        if isinstance(results, dict) and 'detections' in results:
            # Some models return a dictionary with a 'detections' key
            raw_detections = results['detections']
            
            for det in raw_detections:
                if 'bbox' in det and 'confidence' in det and 'class_id' in det:
                    x1, y1, x2, y2 = det['bbox']
                    score = det['confidence']
                    class_id = det['class_id']
                    
                    # Only accept detections for cats or dogs
                    if class_id in CLASSES_TO_DETECT and score >= DETECTION_THRESHOLD:
                        # Convert to integers and ensure within frame bounds
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        x1 = max(0, min(FRAME_WIDTH-1, x1))
                        y1 = max(0, min(FRAME_HEIGHT-1, y1))
                        x2 = max(0, min(FRAME_WIDTH-1, x2))
                        y2 = max(0, min(FRAME_HEIGHT-1, y2))
                        
                        detections.append((x1, y1, x2, y2, score, class_id))
            
            return detections
                
        # Standard DeGirum output format
        if hasattr(results, 'results') and results.results:
            print("Processing standard DeGirum results format")
            
            # Print each result structure for debugging
            for i, output in enumerate(results.results):
                print(f"Output {i} keys: {output.keys() if isinstance(output, dict) else 'not a dict'}")
                if isinstance(output, dict) and 'data' in output:
                    data_shape = output['data'].shape if hasattr(output['data'], 'shape') else 'no shape'
                    print(f"  Data shape: {data_shape}")
                    if hasattr(output['data'], 'shape') and len(output['data'].shape) > 0:
                        print(f"  First few data elements: {output['data'].flatten()[:10]}")
            
            # Some models output detection boxes directly
            for output in results.results:
                if 'detections' in output:
                    for det in output['detections']:
                        if 'bbox' in det and 'score' in det and 'class_id' in det:
                            x1, y1, x2, y2 = det['bbox']
                            score = det['score']
                            class_id = det['class_id']
                            
                            if class_id in CLASSES_TO_DETECT and score >= DETECTION_THRESHOLD:
                                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                                detections.append((x1, y1, x2, y2, score, class_id))
            
            # YOLO-style output format
            for output in results.results:
                if isinstance(output, dict) and 'data' in output:
                    data = output['data']
                    
                    # Handle different data formats
                    if isinstance(data, np.ndarray):
                        # Handle different array shapes
                        if len(data.shape) == 4:  # [batch, grid_h, grid_w, values]
                            data = data.reshape(-1, data.shape[-1])
                        elif len(data.shape) == 3:  # [grid_h, grid_w, values]
                            data = data.reshape(-1, data.shape[-1])
                        
                        # Different models may have different formats for the data
                        # Try parsing as: [x_center, y_center, width, height, objectness, class_scores...]
                        if data.shape[1] >= 6:  # Need at least 6 values per detection
                            for box_data in data:
                                # Check if there are class scores
                                objectness = box_data[4]
                                
                                if objectness >= DETECTION_THRESHOLD:
                                    # Get class ID and score
                                    class_scores = box_data[5:]
                                    class_id = np.argmax(class_scores)
                                    score = float(class_scores[class_id])
                                    
                                    # Only process if it's a cat or dog with high enough confidence
                                    if class_id in CLASSES_TO_DETECT and score >= DETECTION_THRESHOLD:
                                        # Convert center/width/height to x1,y1,x2,y2
                                        x_center, y_center, width, height = box_data[0:4]
                                        
                                        # Convert normalized coordinates (0-1) if needed
                                        if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1:
                                            x_center *= FRAME_WIDTH
                                            y_center *= FRAME_HEIGHT
                                            width *= FRAME_WIDTH
                                            height *= FRAME_HEIGHT
                                        
                                        # Calculate corner coordinates
                                        x1 = int(x_center - width/2)
                                        y1 = int(y_center - height/2)
                                        x2 = int(x_center + width/2)
                                        y2 = int(y_center + height/2)
                                        
                                        # Enforce image boundaries
                                        x1 = max(0, x1)
                                        y1 = max(0, y1)
                                        x2 = min(FRAME_WIDTH, x2)
                                        y2 = min(FRAME_HEIGHT, y2)
                                        
                                        detections.append((x1, y1, x2, y2, score, class_id))
                        
                        # Alternative format: each row is already [x1, y1, x2, y2, score, class_id]
                        elif data.shape[1] == 6:
                            for box_data in data:
                                x1, y1, x2, y2, score, class_id = box_data
                                
                                # Convert class_id to int and score to float
                                class_id = int(class_id)
                                score = float(score)
                                
                                if class_id in CLASSES_TO_DETECT and score >= DETECTION_THRESHOLD:
                                    # Convert to pixel values if normalized
                                    if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
                                        x1 *= FRAME_WIDTH
                                        y1 *= FRAME_HEIGHT
                                        x2 *= FRAME_WIDTH
                                        y2 *= FRAME_HEIGHT
                                    
                                    # Convert to integers and ensure within frame bounds
                                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                                    x1 = max(0, min(FRAME_WIDTH-1, x1))
                                    y1 = max(0, min(FRAME_HEIGHT-1, y1))
                                    x2 = max(0, min(FRAME_WIDTH-1, x2))
                                    y2 = max(0, min(FRAME_HEIGHT-1, y2))
                                    
                                    detections.append((x1, y1, x2, y2, score, class_id))
        
    except Exception as e:
        print(f"Error processing detection results: {e}")
        import traceback
        traceback.print_exc()
    
    # Print the number of valid detections found
    if detections:
        print(f"Found {len(detections)} valid detections in the results")
    
    return detections

def activate_relay(pin, duration=0.1):
    """Activate a relay for the specified duration"""
    if not DEV_MODE:
        try:
            # Debug output
            print(f"Activating relay on pin {pin} for {duration:.2f} seconds")
            
            # Turn ON the relay
            if pin == RELAY_PINS['squirt']:
                # Squirt relay: HIGH = ON, LOW = OFF
                GPIO.output(pin, SQUIRT_RELAY_ON_STATE)
                print(f"  Set squirt relay to ON state: {SQUIRT_RELAY_ON_STATE}")
            else:
                # Standard relays: active-low behavior (LOW = ON, HIGH = OFF)
                active_state = GPIO.LOW if RELAY_ACTIVE_LOW else GPIO.HIGH
                GPIO.output(pin, active_state)
                print(f"  Set relay to active state: {active_state}")
            
            # Wait for the specified duration
            time.sleep(duration)
            
            # Turn OFF the relay
            if pin == RELAY_PINS['squirt']:
                GPIO.output(pin, SQUIRT_RELAY_OFF_STATE)
                print(f"  Set squirt relay to OFF state: {SQUIRT_RELAY_OFF_STATE}")
            else:
                inactive_state = GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW
                GPIO.output(pin, inactive_state)
                print(f"  Set relay to inactive state: {inactive_state}")
                
            return True
        except Exception as e:
            print(f"Error activating relay {pin}: {e}")
            return False
    else:
        # In dev mode, just simulate the relay activation
        print(f"Simulating relay {pin} activation for {duration:.2f} seconds")
        time.sleep(duration)
        return True

def test_relays():
    """Test all relays to ensure they're working properly"""
    if DEV_MODE:
        print("Cannot test relays in development mode")
        return
        
    print("\nTesting all relays...")
    
    # Test each relay with a short activation
    for name, pin in RELAY_PINS.items():
        print(f"Testing {name} relay (pin {pin})...")
        activate_relay(pin, 0.5)
        time.sleep(0.5)  # Wait between relay activations
    
    print("Relay test complete\n")

def calculate_object_zone(detections, frame_width):
    """Calculate which zone the detected object is in"""
    if not detections:
        return None
    
    # Find the detection with highest confidence
    primary_detection = max(detections, key=lambda d: d[4])
    x1, y1, x2, y2, score, class_id = primary_detection
    
    # Calculate center point of detection
    center_x = (x1 + x2) / 2
    
    # Calculate relative position (-1.0 to 1.0 range)
    # We multiply by 2 to get the full -1 to 1 range
    relative_position = ((center_x / frame_width) - 0.5) * 2
    
    # Determine which zone the object is in
    for zone_name, zone_data in TRACKING_ZONES.items():
        min_pos, max_pos = zone_data['range']
        if min_pos <= relative_position <= max_pos:
            return zone_name, relative_position
    
    # Failsafe for out-of-bounds values
    if relative_position < -1.0:
        return 'far_left', relative_position
    else:
        return 'far_right', relative_position

def handle_detection(detections, frame_width, base_activation_duration):
    """Handle detection and calculate appropriate movement using zone-based approach"""
    global CURRENT_ZONE, LAST_DETECTION_TIME, LAST_MOVEMENT_TIME
    
    # Optionally trigger the squirt relay on every detection for testing
    TEST_SQUIRT = False  # Set to True to test the squirt relay
    if TEST_SQUIRT and detections:
        print(f"TEST MODE: Triggering squirt relay for {RELAY_SQUIRT_DURATION:.2f} seconds")
        activate_relay(RELAY_PINS['squirt'], RELAY_SQUIRT_DURATION)
    
    # Calculate object zone based on detections
    zone_info = calculate_object_zone(detections, frame_width)
    
    if zone_info is not None:
        detected_zone, relative_position = zone_info
        LAST_DETECTION_TIME = time.time()
        
        # Only change zones if it's different and cooldown period has passed
        if detected_zone != CURRENT_ZONE and time.time() - LAST_MOVEMENT_TIME >= MOVEMENT_COOLDOWN:
            print(f"Object detected in {detected_zone} zone (position: {relative_position:.2f})")
            
            # Get the relay and duration for this zone
            zone_data = TRACKING_ZONES[detected_zone]
            relay_name = zone_data['relay']
            # Use the fixed duration from the zone definition, not the base duration from the test phase
            duration = zone_data['duration']
            
            # Take action if needed
            if relay_name and duration > 0:
                print(f"Moving {relay_name} for {duration:.2f}s (zone {detected_zone})")
                activate_relay(RELAY_PINS[relay_name], duration)
                LAST_MOVEMENT_TIME = time.time()
            
            # Update current zone
            CURRENT_ZONE = detected_zone
            
            return relay_name if relay_name else 'center', relative_position
        
        return 'no_change', relative_position
    else:
        # Check if we need to reset position due to inactivity
        if time.time() - LAST_DETECTION_TIME > POSITION_RESET_TIME:
            if CURRENT_ZONE != 'center':
                print(f"No detection for {POSITION_RESET_TIME:.1f}s, resetting to center")
                CURRENT_ZONE = 'center'
                return 'reset', 0.0
        
        return 'none', None

def draw_tracking_info(frame, detections, phase, activation_duration, action=None, relative_position=None):
    """Draw tracking information on the frame"""
    height, width = frame.shape[:2]
    
    # Draw center line
    center_x = width // 2
    cv2.line(frame, (center_x, 0), (center_x, height), (0, 255, 255), 2)
    
    # Draw zone dividers
    for zone_name, zone_data in TRACKING_ZONES.items():
        if zone_name != 'far_right':  # Skip last zone boundary
            # Convert zone boundary to pixel position
            _, max_pos = zone_data['range']
            boundary_x = int(center_x + (width / 2) * max_pos)
            
            # Different color for center zone boundaries
            if zone_name == 'left' or zone_name == 'right':
                color = (0, 200, 0)  # Green for center zone
                thickness = 2
            else:
                color = (0, 0, 200)  # Red for outer zones
                thickness = 1
                
            cv2.line(frame, (boundary_x, 0), (boundary_x, height), color, thickness)
    
    # Draw position indicator
    if relative_position is not None:
        # Calculate position marker on screen
        position_x = int(center_x + (width / 2) * relative_position)
        
        # Draw position marker
        cv2.circle(frame, (position_x, height - 30), 8, (0, 255, 255), -1)
        
        # Draw target zone
        cv2.rectangle(frame, (center_x - 20, height - 40), (center_x + 20, height - 20), (0, 255, 0), 2)
    
    # Draw test phase information
    cv2.rectangle(frame, (0, 0), (width, 60), (0, 0, 0), -1)
    phase_text = f"Test Phase: {phase+1}/{NUM_TEST_PHASES}"
    duration_text = f"Base Duration: {activation_duration:.2f}s"
    cv2.putText(frame, phase_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, duration_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw action information if available
    if action:
        action_color = (0, 255, 0) if action == 'center' or action == 'no_change' else (0, 0, 255)
        cv2.rectangle(frame, (width - 300, 0), (width, 60), (0, 0, 0), -1)
        
        # Get current zone info
        zone_text = f"Zone: {CURRENT_ZONE}"
        action_text = f"Action: {action.upper()}"
        
        cv2.putText(frame, zone_text, (width - 290, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, action_text, (width - 290, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_color, 2)
    
    # Draw detections
    for i, detection in enumerate(detections):
        x1, y1, x2, y2, score, class_id = detection
        
        # Get class name
        class_name = COCO_CLASSES.get(class_id, f"Unknown class {class_id}")
        
        # Draw bounding box
        color = COLORS[class_id % len(COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Add label with confidence
        label = f"{class_name}: {score:.2f}"
        cv2.rectangle(frame, (x1, y1-30), (x1+len(label)*15, y1), color, -1)
        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
    
    # Add timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add time since last detection
    time_since_detection = time.time() - LAST_DETECTION_TIME
    if time_since_detection > 5:  # Only show if more than 5 seconds
        time_text = f"No detection for: {time_since_detection:.1f}s"
        cv2.putText(frame, time_text, (width - 250, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    
    return frame

def init_video_writer(phase, activation_duration):
    """Initialize video writer for recording"""
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(VIDEO_OUTPUT_DIR):
            os.makedirs(VIDEO_OUTPUT_DIR)
            print(f"Created video output directory: {VIDEO_OUTPUT_DIR}")
        
        # Generate filename with timestamp and duration
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = os.path.join(
            VIDEO_OUTPUT_DIR, 
            f"tracking_test_phase{phase+1}_duration{activation_duration:.2f}_{timestamp}.mp4"
        )
        
        # Create video writer
        video_writer = cv2.VideoWriter(
            video_filename, 
            VIDEO_CODEC, 
            VIDEO_FPS, 
            (FRAME_WIDTH, FRAME_HEIGHT)
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

def cleanup_resources(camera):
    """Clean up all resources"""
    print("Cleaning up resources...")
    
    # Release camera
    try:
        if camera is not None:
            if DEV_MODE:
                camera.release()
            else:
                camera.stop()
            print("Camera released")
    except Exception as e:
        print(f"Error releasing camera: {e}")
    
    # Clean up GPIO
    if not DEV_MODE:
        try:
            GPIO.cleanup()
            print("GPIO pins cleaned up")
        except Exception as e:
            print(f"Error cleaning up GPIO: {e}")

def signal_handler(sig, frame):
    """Handle interrupt signals"""
    print("\nInterrupt signal received, cleaning up...")
    sys.exit(0)

def main():
    """Main function that runs the tracking tests"""
    global CURRENT_ZONE, LAST_DETECTION_TIME
    
    try:
        # Register signal handler for graceful exit
        signal.signal(signal.SIGINT, signal_handler)
        
        print("\n=== Tracking Calibration Test Script ===\n")
        
        # Check DeGirum installation
        if DEGIRUM_AVAILABLE:
            print("DeGirum module is available")
            try:
                # Check if we can list available models
                print("Checking DeGirum configuration...")
                hosts = dg.list_host_addresses()
                print(f"Available host addresses: {hosts}")
                
                # Check for local Hailo device
                if "@local" in hosts:
                    print("Local Hailo device detected")
                    try:
                        # Try to list models to verify connection
                        models = dg.list_models(inference_host_address="@local")
                        print(f"Available models: {models}")
                    except Exception as e:
                        print(f"Error listing models: {e}")
                else:
                    print("WARNING: No local Hailo device detected")
            except Exception as e:
                print(f"Error checking DeGirum configuration: {e}")
        else:
            print("WARNING: DeGirum module is NOT available")
            print("Object detection will not work without DeGirum")
            print("Please install DeGirum package and ensure Hailo device is connected")
        
        # Setup camera
        camera = setup_camera()
        
        # Setup GPIO if not in dev mode
        if not DEV_MODE:
            setup_gpio()
            # Test relays to ensure they're working properly
            test_relays()
        
        # Load model
        model = load_model()
        if model is None:
            print("WARNING: No object detection model available. Detection will not work.")
            print("This test will record video but won't detect objects or trigger relays.")
            print("Press Ctrl+C to stop the test if you want to fix the DeGirum setup first.")
            response = input("Do you want to continue without object detection? (y/n): ")
            if response.lower() != 'y':
                print("Test aborted. Please fix DeGirum setup and try again.")
                return
        else:
            print(f"Model input shape: {model.input_shape if hasattr(model, 'input_shape') else 'unknown'}")
            if hasattr(model, 'output_shape'):
                print(f"Model output shape: {model.output_shape}")
        
        # Reset tracking variables
        CURRENT_ZONE = 'center'
        LAST_DETECTION_TIME = time.time()
        
        # Run tests for each phase
        for phase in range(NUM_TEST_PHASES):
            # Calculate activation duration for this phase
            activation_duration = BASE_DURATION + (phase * INCREMENT)
            print(f"\n=== Starting Test Phase {phase+1}/{NUM_TEST_PHASES} ===")
            print(f"Base activation duration: {activation_duration:.2f} seconds")
            
            # Initialize video writer for this phase
            video_writer = init_video_writer(phase, activation_duration)
            
            # Initialize phase start time
            phase_start_time = time.time()
            
            # Track frame count for skipping frames
            frame_count = 0
            frame_skip = 3  # Process every Nth frame for inference
            
            # Main loop for this phase
            while time.time() - phase_start_time < TEST_DURATION:
                # Get frame from camera
                frame = get_frame(camera)
                orig_frame = frame.copy()
                
                # Increment frame counter
                frame_count += 1
                
                # Run detection (only on every Nth frame)
                detections = []
                if model is not None and frame_count % frame_skip == 0:
                    try:
                        start_time = time.time()
                        
                        # Use the same approach as in main.py
                        if DEV_MODE:
                            # In dev mode, use direct model call (not relevant for our Raspberry Pi setup)
                            results = model(frame)
                        else:
                            # In production mode, use predict_batch method on the original frame
                            # This is the key line that matches main.py's approach
                            results_generator = model.predict_batch([frame])
                            results = next(results_generator)
                            
                        inference_time = time.time() - start_time
                        print(f"Inference time: {inference_time:.3f}s")
                        
                        # Process detection results using same approach as main.py
                        detections = []
                
                        # Check if results is in expected format
                        if hasattr(results, 'results') and results.results:
                            result_list = results.results
                            print(f"Processing {len(result_list)} results from results.results")
                        elif hasattr(results, 'results') and isinstance(results.results, list):
                            result_list = results.results
                            print(f"Processing {len(result_list)} results from results.results list")
                        elif isinstance(results, list):
                            result_list = results
                            print(f"Processing {len(result_list)} results from direct results list")
                        else:
                            # No clear list of results found
                            result_list = []
                            print(f"No results found. Results type: {type(results)}")
                            if hasattr(results, '__dict__'):
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
                                
                                # Filter for cat and dog classes only
                                if class_id not in CLASSES_TO_DETECT:
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
                                        print(f"Added detection: bbox={x1},{y1},{x2},{y2}, score={score:.2f}, class_id={class_id}")
                            
                            except Exception as e:
                                print(f"Error processing detection: {e}")
                        
                        if detections:
                            detected_classes = [COCO_CLASSES.get(d[5], f'unknown-{d[5]}') for d in detections]
                            print(f"Detected {len(detections)} objects: {detected_classes}")
                        else:
                            print("No detections found above threshold")
                        
                    except Exception as e:
                        print(f"Error during detection: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Process detections with zone-based approach
                if detections:
                    # Handle detection using zone-based approach
                    action, relative_position = handle_detection(detections, FRAME_WIDTH, activation_duration)
                else:
                    # Still check for position reset even without detections
                    action, relative_position = handle_detection([], FRAME_WIDTH, activation_duration)
                
                # Draw tracking information on frame using zone visualization
                annotated_frame = draw_tracking_info(
                    orig_frame.copy(), detections, phase, activation_duration, action, relative_position
                )
                
                # Record frame if video writer is available
                if video_writer is not None:
                    video_writer.write(annotated_frame)
                
                # Display frame if not in headless mode and display is available
                if 'DISPLAY' in os.environ:
                    cv2.imshow("Tracking Test", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Small delay to control loop speed
                time.sleep(0.01)
            
            # End of phase
            print(f"Test Phase {phase+1} completed")
            
            # Clean up video writer
            if video_writer is not None:
                cleanup_video_writer(video_writer)
                video_writer = None
        
        # All phases completed
        print("\n=== All test phases completed ===")
        print(f"Test videos saved in {VIDEO_OUTPUT_DIR}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up all resources
        if 'camera' in locals() and camera is not None:
            cleanup_resources(camera)
        if 'video_writer' in locals() and video_writer is not None:
            cleanup_video_writer(video_writer)

if __name__ == "__main__":
    main() 