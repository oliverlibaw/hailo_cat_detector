#!/usr/bin/env python3
"""
cat_detector.py
A script to detect cats and trigger a water gun to squirt them.
Uses the same tracking approach as the original script but with YOLOv11s COCO model.
"""

import os
import cv2
import time
import numpy as np
import datetime
import signal
import sys
import traceback
import RPi.GPIO as GPIO
from picamera2 import Picamera2
import degirum as dg
import threading
import select

# Import relay configuration
try:
    from relay_config import RELAY_PINS, RELAY_ACTIVE_LOW, SQUIRT_RELAY_ON_STATE, SQUIRT_RELAY_OFF_STATE
    from relay_config import PD_CENTER_THRESHOLD, PD_KP, PD_KD, PD_MIN_PULSE, PD_MAX_PULSE, PD_MOVEMENT_COOLDOWN
    print("Loaded relay configuration from relay_config.py")
except ImportError:
    print("Using default relay configuration (relay_config.py not found)")
    # Default configuration if relay_config.py is not found
    RELAY_PINS = {
        'squirt': 5,  # Squirt relay
        'right': 13,  # Right relay - controls movement to the RIGHT
        'left': 6,    # Left relay - controls movement to the LEFT
        'unused': 15  # Unused relay
    }
    RELAY_ACTIVE_LOW = True
    PD_CENTER_THRESHOLD = 0.12  # Increased from 0.08 to allow more movement
    PD_KP = 0.25  # Increased from 0.15 for stronger movement
    PD_KD = 0.02
    PD_MIN_PULSE = 0.01
    PD_MAX_PULSE = 0.06  # Increased from 0.04 to allow longer movement pulses
    PD_MOVEMENT_COOLDOWN = 0.8

# Camera Settings
FRAME_WIDTH = 1280  # Use higher resolution for better accuracy
FRAME_HEIGHT = 800  # Adjusted for 16:10 or 16:9 aspect ratio
TARGET_FPS = 30
CAMERA_SETTINGS = {
    "AeEnable": True,  # Auto exposure
    "AwbEnable": True,  # Auto white balance
    "AeExposureMode": 0,
    "AeMeteringMode": 0,
    "ExposureTime": 0,  # Let camera auto-select
    "AnalogueGain": 1.0,  # Start with low gain for less noise
    "Brightness": 0.0,  # Neutral
    "Contrast": 1.2,    # Slightly higher for sharper edges
    "Saturation": 1.2,  # Slightly higher for more color
    "FrameRate": TARGET_FPS,
    "AeConstraintMode": 0,
    "AwbMode": 1,
    "ExposureValue": 0.0,
    "NoiseReductionMode": 2
}

# Video Recording Settings
RECORD_DURATION = 60  # seconds
RECORD_FPS = 30
RECORD_FILENAME = "cat_detection_test.mp4"

# Model Settings
MODEL_NAME = "yolo11s_silu_coco--640x640_quant_hailort_hailo8l_1"
MODEL_ZOO_PATH = "/home/pi5/degirum_model_zoo"
MODEL_INPUT_SIZE = (640, 640)
CAT_CLASS_ID = 15  # COCO class ID for cat

# Tracking Settings
POSITION_RESET_TIME = 10.0  # Reset tracking if no detection for this long
SMOOTHING_FACTOR = 0.3  # Factor for smoothing position changes
MIN_MOVEMENT_THRESHOLD = 0.05  # Minimum error to trigger movement
MOVEMENT_COOLDOWN = 0.8  # Cooldown between movements
MIDDLE_THIRD_FACTOR = 0.5  # Use middle half of object for centering

# Squirt Settings
SQUIRT_DURATION = 0.2  # Duration of squirt in seconds
SQUIRT_COOLDOWN = 1.0  # Minimum time between squirts

# Colors for visualization
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_BLUE = (255, 0, 0)  # BGR format

# Global State Variables
last_action = "Initializing"
last_action_time = 0.0
last_detection_time = 0.0
last_squirt_time = 0.0
previous_error = 0.0
last_movement_time = 0.0
last_position = None
position_history = []
running = True
camera = None
model = None
gpio_available = False  # Flag to track if GPIO is available

def get_user_confidence_threshold():
    """Prompt user for confidence threshold."""
    while True:
        try:
            threshold = float(input("Enter minimum confidence threshold (0.0 to 1.0): "))
            if 0.0 <= threshold <= 1.0:
                return threshold
            print("Please enter a value between 0.0 and 1.0")
        except ValueError:
            print("Please enter a valid number")

def setup_gpio():
    """Initialize GPIO pins for relays."""
    try:
        # First cleanup any existing GPIO state
        try:
            GPIO.cleanup()
        except:
            pass
        time.sleep(0.5)  # Longer delay to ensure cleanup is complete
        
        # Set GPIO mode and disable warnings
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Initialize all relay pins as outputs with initial state
        for name, pin in RELAY_PINS.items():
            try:
                # Try to read the pin first to check if it's available
                GPIO.setup(pin, GPIO.IN)
                time.sleep(0.1)
                # If we can read the pin, set it as output
                GPIO.setup(pin, GPIO.OUT)
                # Set initial state
                if name == 'squirt':
                    GPIO.output(pin, SQUIRT_RELAY_OFF_STATE)
                else:
                    off_state = GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW
                    GPIO.output(pin, off_state)
                print(f"Initialized {name} relay on pin {pin}")
            except Exception as e:
                print(f"Warning: Failed to initialize {name} relay on pin {pin}: {e}")
                continue

        print("GPIO Initialized.")
        print(f"Relay Pins: {RELAY_PINS}")
        print(f"Relays Active Low: {RELAY_ACTIVE_LOW}")
        return True

    except Exception as e:
        print(f"Error initializing GPIO: {e}")
        print("Attempting to continue without GPIO control")
        return False

def setup_camera():
    """Setup PiCamera2 for capture."""
    print("Setting up Pi camera...")
    try:
        # First try to cleanup any existing camera processes
        try:
            os.system("sudo pkill -f libcamera")
            time.sleep(1.0)
        except:
            pass
            
        picam2 = Picamera2()
        
        # Try to get the camera configuration
        try:
            camera_config = picam2.create_video_configuration(
                main={
                    "size": (FRAME_WIDTH, FRAME_HEIGHT),
                    "format": "XBGR8888"
                },
                controls=CAMERA_SETTINGS
            )
        except Exception as e:
            print(f"Warning: Failed to create camera configuration: {e}")
            # Try with default configuration
            camera_config = None
        
        # Configure the camera
        try:
            picam2.configure(camera_config)
        except Exception as e:
            print(f"Warning: Failed to configure camera with custom settings: {e}")
            # Try with default configuration
            picam2.configure()
        
        # Start the camera
        picam2.start()
        time.sleep(2.0)  # Allow camera to stabilize
        
        print(f"Pi camera configured: {FRAME_WIDTH}x{FRAME_HEIGHT} @ {TARGET_FPS} FPS")
        return picam2
        
    except Exception as e:
        print(f"CRITICAL: Error setting up Pi camera: {e}")
        raise

def load_model(model_name, zoo_path, confidence_threshold):
    """Load the specified Degirum model for Hailo hardware acceleration."""
    try:
        print(f"Loading Degirum model: {model_name} from {zoo_path}")
        
        # Create logs directory for HailoRT logs
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        os.chmod(log_dir, 0o777)
        
        log_path = os.path.join(os.path.abspath(os.getcwd()), log_dir)
        os.environ["HAILORT_LOG_PATH"] = log_path
        
        if not os.path.exists(zoo_path):
            print(f"ERROR: Model zoo path not found: {zoo_path}")
            return None

        # Add retry logic for model loading
        max_retries = 3
        retry_delay = 2.0  # seconds
        
        for attempt in range(max_retries):
            try:
                loaded_model = dg.load_model(
                    model_name=model_name,
                    inference_host_address="@local",
                    zoo_url=zoo_path,
                    output_confidence_threshold=confidence_threshold
                )
                
                print(f"Model '{model_name}' loaded successfully.")
                return loaded_model
                
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed to load model: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("All retry attempts failed.")
                    return None

    except Exception as e:
        print(f"CRITICAL: Failed to load model: {e}")
        return None

def activate_relay(pin, duration):
    """Activate a relay for a specified duration."""
    global last_action, last_action_time
    
    try:
        # Determine ON/OFF states
        if pin == RELAY_PINS['squirt']:
            on_state = SQUIRT_RELAY_ON_STATE
            off_state = SQUIRT_RELAY_OFF_STATE
        else:
            on_state = GPIO.LOW if RELAY_ACTIVE_LOW else GPIO.HIGH
            off_state = GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW
        
        # Activate relay
        GPIO.output(pin, on_state)
        time.sleep(duration)
        GPIO.output(pin, off_state)
        
        # Update action tracking
        relay_name = [name for name, p in RELAY_PINS.items() if p == pin][0]
        last_action = f"{relay_name.upper()} ON ({duration:.3f}s)"
        last_action_time = time.time()
        
    except Exception as e:
        print(f"Error activating relay: {e}")

def handle_tracking(bbox, frame_width):
    """
    Handle object tracking with improved movement control.
    Uses the middle half of the object for more stable centering.
    Returns the current error for visualization.
    """
    global previous_error, last_movement_time, last_action, last_action_time
    global last_detection_time, last_position, position_history, last_squirt_time

    x1, y1, x2, y2 = map(int, bbox)
    current_time = time.time()
    
    # Calculate object width and middle half
    object_width = x2 - x1
    middle_half_width = object_width * MIDDLE_THIRD_FACTOR
    middle_half_start = x1 + (object_width - middle_half_width) / 2
    middle_half_end = middle_half_start + middle_half_width
    
    # Calculate center of middle half
    middle_half_center = (middle_half_start + middle_half_end) / 2
    
    # Calculate normalized center position (0-1)
    normalized_center = middle_half_center / frame_width
    
    # Apply smoothing to position changes
    if last_position is not None:
        smoothed_position = last_position + SMOOTHING_FACTOR * 0.5 * (normalized_center - last_position)
        normalized_center = smoothed_position
    
    # Track position history
    position_entry = {
        'time': current_time,
        'position': normalized_center,
        'bbox': bbox
    }
    position_history.append(position_entry)
    if len(position_history) > 20:
        position_history.pop(0)
    
    # Update last known position
    last_position = normalized_center
    last_detection_time = current_time

    # Calculate normalized error (-1.0 to 1.0)
    current_error = ((middle_half_center / frame_width) - 0.5) * 2
    
    # Calculate the derivative component for smoother transitions
    error_derivative = current_error - previous_error
    
    # Update previous error
    previous_error = current_error
    
    # Define tracking zones
    zones = {
        'left': {'range': (-1.0, -PD_CENTER_THRESHOLD), 'relay': 'right', 'action': 'MOVE RIGHT'},
        'center': {'range': (-PD_CENTER_THRESHOLD, PD_CENTER_THRESHOLD), 'relay': None, 'action': 'CENTER'},
        'right': {'range': (PD_CENTER_THRESHOLD, 1.0), 'relay': 'left', 'action': 'MOVE LEFT'}
    }
    
    # Determine current zone
    current_zone = None
    for zone, config in zones.items():
        low, high = config['range']
        if low <= current_error <= high:
            current_zone = zone
            break
    
    if current_zone is None:
        current_zone = 'center'
    
    # Check if we need to move
    if current_zone != 'center' and (current_time - last_movement_time >= MOVEMENT_COOLDOWN):
        # Only move if error is significant enough
        if abs(current_error) > MIN_MOVEMENT_THRESHOLD:
            relay_name = zones[current_zone]['relay']
            action_desc = zones[current_zone]['action']
            
            if relay_name:
                # Calculate pulse duration using PD control
                base_duration = PD_KP * abs(current_error) * 0.2
                derivative_adjustment = PD_KD * error_derivative * (1 if current_error > 0 else -1) * 0.2
                pulse_duration = base_duration + derivative_adjustment
                
                # Clamp to min/max values
                pulse_duration = max(PD_MIN_PULSE, min(pulse_duration, PD_MAX_PULSE))
                
                # Add additional safety check for very small movements
                if pulse_duration > PD_MIN_PULSE:
                    # Activate the relay
                    print(f"PD {action_desc} (Error: {current_error:.3f}, Duration: {pulse_duration:.3f}s)")
                    activate_relay(RELAY_PINS[relay_name], pulse_duration)
                    last_movement_time = current_time
    
    # Check if we should squirt (only in center zone)
    if current_zone == 'center' and (current_time - last_squirt_time >= SQUIRT_COOLDOWN):
        print("Target centered! Activating squirt relay...")
        activate_relay(RELAY_PINS['squirt'], SQUIRT_DURATION)
        last_squirt_time = current_time
    
    return current_error

def draw_frame_elements(frame, fps, detections, current_error=None):
    """Draw tracking information on the frame."""
    height, width = frame.shape[:2]
    display_frame = frame.copy()
    
    # Draw center line
    center_x = width // 2
    cv2.line(display_frame, (center_x, 0), (center_x, height), COLOR_YELLOW, 1)
    
    # Draw zone boundaries
    left_boundary = int(center_x - (width / 2) * PD_CENTER_THRESHOLD)
    right_boundary = int(center_x + (width / 2) * PD_CENTER_THRESHOLD)
    cv2.line(display_frame, (left_boundary, height - 25), (left_boundary, height - 15), COLOR_GREEN, 2)
    cv2.line(display_frame, (right_boundary, height - 25), (right_boundary, height - 15), COLOR_GREEN, 2)
    
    # Draw detections
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        score = det['score']
        label = det['label']
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        
        # Draw bounding box with thicker line
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), COLOR_BLUE, 3)
        
        # Draw label with confidence
        text = f"{label}: {score:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = x1
        text_y = y1 - 10 if y1 > 30 else y1 + 30  # Adjust text position if too close to top
        
        # Draw text background with padding
        padding = 5
        cv2.rectangle(display_frame, 
                     (text_x - padding, text_y - text_size[1] - padding),
                     (text_x + text_size[0] + padding, text_y + padding),
                     COLOR_BLUE, -1)
        
        # Draw text
        cv2.putText(display_frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)
    
    # Draw error indicator
    if current_error is not None:
        marker_pos_x = int(center_x + (width / 2) * current_error)
        cv2.circle(display_frame, (marker_pos_x, height - 40), 8, COLOR_RED, -1)
        cv2.line(display_frame, (marker_pos_x, height-40), (marker_pos_x, height-20), COLOR_RED, 2)
        
        # Display error value and movement status
        error_text = f"Error: {current_error:.3f}"
        cv2.putText(display_frame, error_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
        
        # Show movement status with larger text
        if abs(current_error) > PD_CENTER_THRESHOLD:
            if current_error > 0:
                move_text = "MOVE LEFT"
                text_color = COLOR_RED
            else:
                move_text = "MOVE RIGHT"
                text_color = COLOR_RED
            
            # Draw movement text in center of screen with large font
            text_size = cv2.getTextSize(move_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height // 2
            cv2.putText(display_frame, move_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, text_color, 3)
    
    # Draw FPS
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(display_frame, fps_text, (width - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)
    
    # Draw last action with larger text if recent
    if time.time() - last_action_time < 2.0:
        action_text = f"Last: {last_action}"
        text_size = cv2.getTextSize(action_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height - 50
        
        # Draw text background
        padding = 10
        cv2.rectangle(display_frame,
                     (text_x - padding, text_y - text_size[1] - padding),
                     (text_x + text_size[0] + padding, text_y + padding),
                     COLOR_YELLOW, -1)
        
        # Draw text
        cv2.putText(display_frame, action_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 2)
        
        # If it's a squirt action, show large SQUIRT text
        if "SQUIRT" in last_action:
            squirt_text = "SQUIRT!"
            text_size = cv2.getTextSize(squirt_text, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 4)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height // 3
            cv2.putText(display_frame, squirt_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.0, COLOR_RED, 4)
    
    return display_frame

def cleanup():
    """Clean up resources before exiting."""
    print("\nCleaning up...")
    global running, camera
    
    running = False
    
    if camera is not None:
        try:
            camera.stop()
            print("Camera stopped.")
        except Exception as e:
            print(f"Error stopping camera: {e}")
    
    # Turn off all relays only if GPIO is available
    try:
        for name, pin in RELAY_PINS.items():
            try:
                if name == 'squirt':
                    GPIO.output(pin, SQUIRT_RELAY_OFF_STATE)
                else:
                    off_state = GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW
                    GPIO.output(pin, off_state)
            except:
                pass
        GPIO.cleanup()
        print("GPIO cleaned up.")
    except Exception as e:
        print(f"Error during GPIO cleanup: {e}")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global running
    print("\nCtrl+C detected. Exiting gracefully...")
    running = False

def main():
    """Main function for cat detection and squirting."""
    global running, camera, model, previous_error
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n=== Cat Detection and Squirting System ===")
    print("Press Ctrl+C to exit.")
    
    # Get user input for confidence threshold
    DETECTION_THRESHOLD = get_user_confidence_threshold()
    print(f"Using confidence threshold: {DETECTION_THRESHOLD}")
    
    try:
        # Initialize hardware
        gpio_initialized = setup_gpio()
        if not gpio_initialized:
            print("Warning: Continuing without GPIO control")
        
        # Try to setup camera multiple times if needed
        max_camera_attempts = 3
        for attempt in range(max_camera_attempts):
            try:
                camera = setup_camera()
                # Wait for camera to stabilize
                time.sleep(2.0)
                break
            except Exception as e:
                print(f"Camera setup attempt {attempt + 1}/{max_camera_attempts} failed: {e}")
                if attempt < max_camera_attempts - 1:
                    print("Retrying in 2 seconds...")
                    time.sleep(2.0)
                else:
                    print("Failed to setup camera after all attempts")
                    return
        
        # Load model
        model = load_model(MODEL_NAME, MODEL_ZOO_PATH, DETECTION_THRESHOLD)
        if model is None:
            print("CRITICAL: Failed to load model.")
            print("Will exit in 3 seconds...")
            time.sleep(3)
            return
        
        # Print model information
        try:
            if hasattr(model, 'model_name'):
                print(f"Model name: {model.model_name}")
            if hasattr(model, 'input_shape'):
                print(f"Model input shape: {model.input_shape}")
            if hasattr(model, 'output_names'):
                print(f"Model output names: {model.output_names}")
            print(f"Detection threshold: {DETECTION_THRESHOLD}")
            print(f"Target class: cat (COCO class {CAT_CLASS_ID})")
        except Exception as e:
            print(f"Warning: Could not print model information: {e}")

        # Initialize FPS counter
        fps_counter = FPSCounter()
        last_fps_print_time = time.time()
        
        # Initialize video recording
        print(f"Starting video recording for {RECORD_DURATION} seconds...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(RECORD_FILENAME, fourcc, RECORD_FPS, (FRAME_WIDTH, FRAME_HEIGHT))
        recording_start_time = time.time()
        
        while running:
            try:
                # Check if recording duration has elapsed
                if time.time() - recording_start_time >= RECORD_DURATION:
                    print(f"Recording complete: {RECORD_FILENAME}")
                    video_writer.release()
                    running = False
                    break
                
                # Read frame
                frame = camera.capture_array()
                if frame.shape[2] == 4:  # Convert XBGR to BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                
                # Run inference
                detections = []
                try:
                    # Preprocess frame
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = model.predict_batch([rgb_frame])
                    result = next(results)
                    
                    # Process detections
                    if hasattr(result, 'results'):
                        for det in result.results:
                            if det['score'] >= DETECTION_THRESHOLD and det['category_id'] == CAT_CLASS_ID:
                                # Convert bbox to proper format (x1, y1, x2, y2)
                                bbox = det['bbox']
                                if isinstance(bbox, dict):
                                    x1, y1, x2, y2 = bbox['left'], bbox['top'], bbox['right'], bbox['bottom']
                                elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                                    x1, y1, x2, y2 = bbox
                                else:
                                    print(f"Warning: Unknown bbox format: {bbox}")
                                    continue
                                
                                # Ensure coordinates are integers
                                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                                
                                detections.append({
                                    'bbox': (x1, y1, x2, y2),
                                    'score': det['score'],
                                    'label': 'cat'
                                })
                except Exception as e:
                    print(f"Error during inference: {e}")
                    # If we get a Hailo device error, try to reload the model
                    if "HAILO_RPC_FAILED" in str(e):
                        print("Hailo device error detected. Attempting to reload model...")
                        model = load_model(MODEL_NAME, MODEL_ZOO_PATH, DETECTION_THRESHOLD)
                        if model is None:
                            print("Failed to reload model. Exiting...")
                            running = False
                            break
                        continue
                
                # Handle tracking
                current_error = None
                if detections:
                    # Use highest confidence detection
                    target_detection = max(detections, key=lambda d: d['score'])
                    current_error = handle_tracking(target_detection['bbox'], FRAME_WIDTH)
                else:
                    # Reset tracking if no detection for a while
                    if time.time() - last_detection_time > POSITION_RESET_TIME:
                        if previous_error != 0.0:
                            print(f"Resetting tracking due to inactivity ({POSITION_RESET_TIME}s)")
                            previous_error = 0.0
                
                # Update FPS
                fps_counter.update()
                current_fps = fps_counter.get_fps()
                if time.time() - last_fps_print_time >= 5.0:
                    print(f"FPS: {current_fps:.1f}")
                    last_fps_print_time = time.time()
                
                # Draw visualization
                display_frame = draw_frame_elements(frame, current_fps, detections, current_error)
                
                # Write frame to video
                video_writer.write(display_frame)
                
                # Show frame
                cv2.imshow("Cat Tracking", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    running = False
                    
            except Exception as e:
                print(f"Error in main loop: {e}")
                if "HAILO_RPC_FAILED" in str(e):
                    print("Hailo device error detected. Attempting to reload model...")
                    model = load_model(MODEL_NAME, MODEL_ZOO_PATH, DETECTION_THRESHOLD)
                    if model is None:
                        print("Failed to reload model. Exiting...")
                        running = False
                        break
                time.sleep(1.0)  # Add delay before retrying
            
    except Exception as e:
        print(f"\nError in main loop: {e}")
        traceback.print_exc()
    finally:
        cleanup()
        if 'video_writer' in locals():
            video_writer.release()
        cv2.destroyAllWindows()

class FPSCounter:
    """Simple FPS counter class."""
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

if __name__ == "__main__":
    main() 