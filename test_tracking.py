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
    DEGIRUM_AVAILABLE = True
except ImportError:
    DEGIRUM_AVAILABLE = False
    print("DeGirum not available, using fallback detection")

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
            model = dg.load_model(
                model_name=model_name,
                inference_host_address="@local",
                zoo_url=zoo_path
            )
        else:
            # Fall back to public model
            print("Local model zoo not found, using yolov8n_coco from public model")
            model = dg.load_model(
                model_name="yolov8n_coco",
                inference_host_address="@local",
                zoo_url="degirum/public"
            )
            
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
        print("Will use OpenCV DNN instead")
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

def handle_detection(bbox, frame_width, activation_duration):
    """Handle detection and activate appropriate relays to center the object"""
    x1, y1, x2, y2 = map(int, bbox)
    center_x = (x1 + x2) / 2
    relative_position = (center_x / frame_width) - 0.5  # -0.5 to 0.5 range
    
    # Optionally trigger the squirt relay on every detection for testing
    TEST_SQUIRT = False  # Set to True to test the squirt relay
    if TEST_SQUIRT:
        print(f"TEST MODE: Triggering squirt relay for {RELAY_SQUIRT_DURATION:.2f} seconds")
        activate_relay(RELAY_PINS['squirt'], RELAY_SQUIRT_DURATION)
    
    # Activate relay based on object position
    if relative_position < -CENTER_THRESHOLD:
        # Object is on the left side, activate right relay to move camera left
        print(f"Object on left side (position: {relative_position:.2f}), moving right for {activation_duration:.2f} seconds")
        activate_relay(RELAY_PINS['right'], activation_duration)
        return 'right', relative_position
    elif relative_position > CENTER_THRESHOLD:
        # Object is on the right side, activate left relay to move camera right
        print(f"Object on right side (position: {relative_position:.2f}), moving left for {activation_duration:.2f} seconds")
        activate_relay(RELAY_PINS['left'], activation_duration)
        return 'left', relative_position
    else:
        # Object is already centered
        print(f"Object centered (position: {relative_position:.2f}), no movement needed")
        return 'center', relative_position

def draw_tracking_info(frame, detections, phase, activation_duration, action=None, relative_position=None):
    """Draw tracking information on the frame"""
    height, width = frame.shape[:2]
    
    # Draw center line
    center_x = width // 2
    cv2.line(frame, (center_x, 0), (center_x, height), (0, 255, 255), 2)
    
    # Draw threshold lines
    left_threshold = int(width * (0.5 - CENTER_THRESHOLD))
    right_threshold = int(width * (0.5 + CENTER_THRESHOLD))
    cv2.line(frame, (left_threshold, 0), (left_threshold, height), (0, 0, 255), 2)
    cv2.line(frame, (right_threshold, 0), (right_threshold, height), (0, 0, 255), 2)
    
    # Draw test phase information
    cv2.rectangle(frame, (0, 0), (width, 60), (0, 0, 0), -1)
    phase_text = f"Test Phase: {phase+1}/{NUM_TEST_PHASES}"
    duration_text = f"Activation Duration: {activation_duration:.2f}s"
    cv2.putText(frame, phase_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, duration_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw action information if available
    if action:
        action_color = (0, 255, 0) if action == 'center' else (0, 0, 255)  # Green for centered, red for moving
        cv2.rectangle(frame, (width - 200, 0), (width, 60), (0, 0, 0), -1)
        action_text = f"Action: {action.upper()}"
        position_text = f"Position: {relative_position:.2f}" if relative_position is not None else ""
        cv2.putText(frame, action_text, (width - 190, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_color, 2)
        cv2.putText(frame, position_text, (width - 190, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
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
    try:
        # Register signal handler for graceful exit
        signal.signal(signal.SIGINT, signal_handler)
        
        print("\n=== Tracking Calibration Test Script ===\n")
        
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
            print("Warning: No object detection model available. Detection will not work.")
        else:
            print(f"Model input shape: {model.input_shape if hasattr(model, 'input_shape') else 'unknown'}")
            if hasattr(model, 'output_shape'):
                print(f"Model output shape: {model.output_shape}")
        
        # Run tests for each phase
        for phase in range(NUM_TEST_PHASES):
            # Calculate activation duration for this phase
            activation_duration = BASE_DURATION + (phase * INCREMENT)
            print(f"\n=== Starting Test Phase {phase+1}/{NUM_TEST_PHASES} ===")
            print(f"Activation duration: {activation_duration:.2f} seconds")
            
            # Initialize video writer for this phase
            video_writer = init_video_writer(phase, activation_duration)
            
            # Initialize phase start time
            phase_start_time = time.time()
            
            # Main loop for this phase
            while time.time() - phase_start_time < TEST_DURATION:
                # Get frame from camera
                frame = get_frame(camera)
                orig_frame = frame.copy()
                
                # Run detection
                detections = []
                if model is not None:
                    try:
                        # Convert BGR to RGB for model input
                        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Prepare model input (match the approach in main.py)
                        # Resize with letterbox to match model input requirements
                        input_frame, scale, pad_top, pad_left = resize_image_letterbox(
                            input_frame, (640, 640)
                        )
                        
                        # Add batch dimension if needed
                        if hasattr(model, 'input_shape') and model.input_shape[0] == 1:
                            input_frame = np.expand_dims(input_frame, axis=0)
                            
                        # Run inference with the model (direct call like in main.py)
                        start_time = time.time()
                        results = model(input_frame)
                        inference_time = time.time() - start_time
                        print(f"Inference time: {inference_time:.3f}s")
                        
                        # Try alternative method with named inputs if previous call fails to detect
                        if hasattr(results, 'results') and len(results.results) > 0:
                            print("Using results from standard model call")
                        else:
                            print("Trying alternative prediction method...")
                            # Some models require named inputs
                            results = model.predict({"input": input_frame})
                            print(f"Alternative method results type: {type(results)}")
                            
                        # Process detection results
                        detections = process_detections(results)
                        
                        if detections:
                            detected_classes = [COCO_CLASSES.get(d[5], f'unknown-{d[5]}') for d in detections]
                            print(f"Detected {len(detections)} objects: {detected_classes}")
                        
                    except Exception as e:
                        print(f"Error during detection: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Initialize action and relative position
                action = None
                relative_position = None
                
                # Process primary detection if any
                if detections:
                    # Find detection with highest confidence
                    primary_detection = max(detections, key=lambda d: d[4])
                    x1, y1, x2, y2, score, class_id = primary_detection
                    
                    # Handle detection (activate relays)
                    action, relative_position = handle_detection((x1, y1, x2, y2), FRAME_WIDTH, activation_duration)
                
                # Draw tracking information on frame
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