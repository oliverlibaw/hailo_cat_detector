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
SQUIRT_RELAY_ON_STATE = 1  # GPIO.HIGH
SQUIRT_RELAY_OFF_STATE = 0  # GPIO.LOW

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
    """Load detection model"""
    if not DEGIRUM_AVAILABLE:
        print("DeGirum not available, using OpenCV DNN instead")
        return None
        
    try:
        print("Loading DeGirum model...")
        model_name = "yolov8n_coco"
        
        # Try to load from local zoo first
        zoo_path = "/home/pi5/degirum_model_zoo"
        if os.path.exists(zoo_path):
            print(f"Loading from local model zoo: {zoo_path}")
            model = dg.load_model(
                model_name=model_name,
                inference_host_address="@local",
                zoo_url=zoo_path,
                output_confidence_threshold=DETECTION_THRESHOLD
            )
        else:
            # Fall back to public model
            print("Local model zoo not found, using public model")
            model = dg.load_model(
                model_name=model_name,
                inference_host_address="@local",
                zoo_url="degirum/public",
                output_confidence_threshold=DETECTION_THRESHOLD
            )
            
        print(f"Model loaded successfully")
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

def process_detections(frame, results, model):
    """Process detection results and return detections list"""
    detections = []
    
    if model is None:
        # Fallback detection not implemented in this script
        return []
        
    try:
        # YOLO model returns results in different formats depending on the version
        if hasattr(results, 'results') and results.results:
            result_list = results.results
        elif hasattr(results, 'results') and isinstance(results.results, list):
            result_list = results.results
        elif isinstance(results, list):
            result_list = results
        else:
            result_list = []
        
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
            
            except Exception as e:
                print(f"Error processing detection: {e}")
        
    except Exception as e:
        print(f"Error processing detection results: {e}")
        import traceback
        traceback.print_exc()
    
    return detections

def activate_relay(pin, duration=0.1):
    """Activate a relay for the specified duration"""
    if not DEV_MODE:
        try:
            # Turn ON the relay
            if pin == RELAY_PINS['squirt']:
                # Squirt relay: HIGH = ON, LOW = OFF
                GPIO.output(pin, SQUIRT_RELAY_ON_STATE)
            else:
                # Standard relays: active-low behavior (LOW = ON, HIGH = OFF)
                GPIO.output(pin, GPIO.LOW if RELAY_ACTIVE_LOW else GPIO.HIGH)
            
            # Wait for the specified duration
            time.sleep(duration)
            
            # Turn OFF the relay
            if pin == RELAY_PINS['squirt']:
                GPIO.output(pin, SQUIRT_RELAY_OFF_STATE)
            else:
                GPIO.output(pin, GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW)
                
            return True
        except Exception as e:
            print(f"Error activating relay {pin}: {e}")
            return False
    else:
        # In dev mode, just simulate the relay activation
        print(f"Simulating relay {pin} activation for {duration:.2f} seconds")
        time.sleep(duration)
        return True

def handle_detection(bbox, frame_width, activation_duration):
    """Handle detection and activate appropriate relays to center the object"""
    x1, y1, x2, y2 = map(int, bbox)
    center_x = (x1 + x2) / 2
    relative_position = (center_x / frame_width) - 0.5  # -0.5 to 0.5 range
    
    # Activate relay based on object position
    if relative_position < -CENTER_THRESHOLD:
        # Object is on the left side, activate right relay to move camera left
        print(f"Object on left side, moving right for {activation_duration:.2f} seconds")
        activate_relay(RELAY_PINS['right'], activation_duration)
        return 'right', relative_position
    elif relative_position > CENTER_THRESHOLD:
        # Object is on the right side, activate left relay to move camera right
        print(f"Object on right side, moving left for {activation_duration:.2f} seconds")
        activate_relay(RELAY_PINS['left'], activation_duration)
        return 'left', relative_position
    else:
        # Object is already centered
        print("Object centered, no movement needed")
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
            print("Cleaning up GPIO...")
            GPIO.cleanup()
            print("GPIO cleaned up")
        except Exception as e:
            print(f"Error cleaning up GPIO: {e}")
    
    print("All resources cleaned up")

def signal_handler(sig, frame):
    """Handle Ctrl+C signal"""
    print("\nProgram terminated by user")
    if 'camera' in globals() and camera is not None:
        cleanup_resources(camera)
    if 'video_writer' in globals() and video_writer is not None:
        cleanup_video_writer(video_writer)
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
        
        # Load model
        model = load_model()
        
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
                
                # Run detection
                if model is not None:
                    results = model.predict_batch([frame])
                    results = next(results)
                    detections = process_detections(frame, results, model)
                else:
                    # Fallback detection not implemented
                    detections = []
                
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
                    frame.copy(), detections, phase, activation_duration, action, relative_position
                )
                
                # Record frame if video writer is available
                if video_writer is not None:
                    video_writer.write(annotated_frame)
                
                # Display frame if not in headless mode
                if not 'DISPLAY' not in os.environ:
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