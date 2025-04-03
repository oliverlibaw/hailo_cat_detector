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
CONFIDENCE_THRESHOLD = 0.2  # Lower confidence threshold to detect more objects
MODEL_INPUT_SIZE = (640, 640)  # YOLOv8n input size
CENTER_THRESHOLD = 0.1  # Threshold for determining if object is left/right of center
RELAY_CENTER_DURATION = 0.2  # Duration to activate center relay
RELAY_CENTER_COOLDOWN = 1.0  # Cooldown period for center relay
INFERENCE_INTERVAL = 0.2  # Run inference every 200ms to reduce load

# Cat class names
CAT_CLASSES = {
    0: "Gary",
    1: "George",
    2: "Fred"
}

# Colors for visualization
COLORS = {
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

# Sound effects for development mode
SOUND_FILES = [
    'cat1.wav', 'cat2.wav', 'cat3.wav', 'cat4.wav', 'cat5.wav', 'cat6.wav'
]

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
    """Set up the camera with proper error handling"""
    if DEV_MODE:
        print("Attempting to access webcam...")
        print("Note: On macOS, you may need to grant camera permissions to Terminal/IDE")
        print("If the camera doesn't work, try running this script from Terminal")
        
        # Try different camera indices
        for i in range(3):  # Try first 3 camera indices
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)  # Use AVFoundation backend on macOS
                if cap.isOpened():
                    print(f"Successfully opened camera {i}")
                    return cap
            except Exception as e:
                print(f"Failed to open camera {i}: {str(e)}")
                continue
        
        raise Exception("Could not open webcam")
    else:
        try:
            print("Setting up Pi camera...")
            picam2 = Picamera2()
            
            # Configure camera
            camera_config = picam2.create_preview_configuration(
                main={"format": 'RGB888', "size": MODEL_INPUT_SIZE}
            )
            picam2.configure(camera_config)
            picam2.start()
            print("Successfully initialized Pi camera")
            return picam2
        except Exception as e:
            print(f"Failed to setup Pi camera: {e}")
            print("Please ensure:")
            print("1. Camera is enabled in raspi-config")
            print("2. You have the necessary permissions")
            print("3. The camera is properly connected")
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
            if hasattr(results, 'results') and results.results:
                print(f"Processing {len(results.results)} detections")
                # Process results attribute
                for detection in results.results:
                    try:
                        # Extract bounding box, score and class ID based on what's available
                        if hasattr(detection, 'bbox'):
                            # If bbox is an object
                            if hasattr(detection.bbox, 'x1'):
                                x1 = int(detection.bbox.x1)
                                y1 = int(detection.bbox.y1)
                                x2 = int(detection.bbox.x2)
                                y2 = int(detection.bbox.y2)
                            # If bbox is a list/tuple
                            else:
                                x1 = int(detection.bbox[0])
                                y1 = int(detection.bbox[1])
                                x2 = int(detection.bbox[2])
                                y2 = int(detection.bbox[3])
                                
                            # Get score and class ID
                            if hasattr(detection, 'confidence'):
                                score = float(detection.confidence)
                            elif hasattr(detection, 'score'):
                                score = float(detection.score)
                            else:
                                score = 0.0
                                
                            if hasattr(detection, 'class_id'):
                                class_id = int(detection.class_id)
                            elif hasattr(detection, 'category_id'):
                                class_id = int(detection.category_id)
                            else:
                                class_id = 0
                                
                            # Print detailed info about this detection
                            cat_name = CAT_CLASSES.get(class_id, f"Unknown class {class_id}")
                            print(f"Detection: {cat_name}, bbox={x1},{y1},{x2},{y2}, score={score:.2f}")
                            
                            detections.append((x1, y1, x2, y2, score, class_id))
                    except Exception as e:
                        print(f"Error processing detection: {e}")
                        # Print detailed error for debugging
                        import traceback
                        traceback.print_exc()
                        
            elif hasattr(results, 'detections'):
                for detection in results.detections:
                    try:
                        # Get bounding box coordinates
                        x1 = int(detection.bbox.x1)
                        y1 = int(detection.bbox.y1)
                        x2 = int(detection.bbox.x2)
                        y2 = int(detection.bbox.y2)
                        
                        # Get score and class ID
                        score = float(detection.confidence)
                        class_id = int(detection.class_id)
                        
                        detections.append((x1, y1, x2, y2, score, class_id))
                    except Exception as e:
                        print(f"Error processing detection: {e}")
            
            # Try yet another method
            elif hasattr(results, 'bboxes'):
                for i in range(len(results.bboxes)):
                    bbox = results.bboxes[i]
                    score = results.scores[i]
                    class_id = results.class_ids[i]
                    x1, y1, x2, y2 = map(int, bbox)
                    detections.append((x1, y1, x2, y2, score, class_id))
        
        except Exception as e:
            print(f"Error in process_detections: {e}")
            import traceback
            traceback.print_exc()
    
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
    """Load the YOLOv8 model for cat detection."""
    try:
        print("Loading Degirum model for Hailo accelerator...")
        import degirum as dg
        
        # Load the model with local inference
        model = dg.load_model(
            model_name="yolov8n_cats",
            inference_host_address="@local",  # Use @local for local inference
            zoo_url="/home/pi5/degirum_model_zoo",  # Path to your model zoo
            output_confidence_threshold=0.2,  # Lower threshold to increase detections
            overlay_font_scale=2.5,  # Font scale for overlay
            overlay_show_probabilities=True  # Show confidence scores
        )
        
        # Print model properties to help with debugging
        print("Model loaded successfully")
        print(f"Model properties: {dir(model)}")
        if hasattr(model, 'info'):
            print(f"Model info: {model.info}")
        
        return model
        
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        print("Detailed error traceback:")
        import traceback
        traceback.print_exc()
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
            cv2.imwrite(filename, save_img)
            print(f"Saved frame to {filename}")
            return True
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
    x1, y1, x2, y2, score, class_id = detection
    
    # Get cat name and color
    cat_name = CAT_CLASSES.get(class_id, f"Unknown-{class_id}")
    color = COLORS.get(cat_name.lower(), COLORS['unknown'])
    
    # Ensure coordinates are valid
    height, width = frame.shape[:2]
    x1 = max(0, min(x1, width-1))
    y1 = max(0, min(y1, height-1))
    x2 = max(0, min(x2, width-1))
    y2 = max(0, min(y2, height-1))
    
    # Draw thick bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    
    # Add filled background for text
    label_text = f"{cat_name}: {score:.2f}"
    text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_w, text_h = text_size
    
    # Draw filled rectangle for text background
    cv2.rectangle(frame, 
                 (x1, y1 - text_h - 10), 
                 (x1 + text_w + 10, y1), 
                 color, -1)  # -1 means filled
    
    # Draw text with contrasting color
    cv2.putText(frame, label_text, 
               (x1 + 5, y1 - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['white'], 2)
    
    return frame

try:
    # Load AI model with proper error handling
    model = load_model()
    setup_sound()
    
    # Setup GPIO and camera
    if not DEV_MODE:
        setup_gpio()
    camera = setup_camera()
    
    # Print model info
    print(f"Model: {model_name}")
    print(f"Model input size: {MODEL_INPUT_SIZE}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Development mode: {'Enabled' if DEV_MODE else 'Disabled'}")
    print("\nPress Ctrl+C to stop the program")
    
    start_time = time.time()
    frame_count = 0
    fps = 0
    last_frame_save = 0
    last_inference_time = 0
    
    print("Starting main loop - press Ctrl+C to exit")
    
    while True:
        try:
            current_time = time.time()
            
            # Get frame from camera
            frame = get_frame(camera)
            frame_width = frame.shape[1]
            
            # Create display frame for saving (without display)
            display_frame = frame.copy()
            
            # Only run inference at specified intervals
            run_inference = (current_time - last_inference_time) >= INFERENCE_INTERVAL
            
            if run_inference:
                # Perform inference
                if DEV_MODE:
                    results = model(frame, conf=CONFIDENCE_THRESHOLD)
                else:
                    # Use Degirum's predict_batch method which returns a generator
                    print(f"\nRunning inference at t={current_time:.1f}s")
                    
                    # Set a timeout for inference
                    inference_start = time.time()
                    inference_timeout = 5.0  # 5 seconds timeout
                    results = None
                    
                    try:
                        prediction_generator = model.predict_batch([frame])
                        # Get the first result from the generator
                        for result in prediction_generator:
                            results = result
                            print(f"Got result type: {type(results)}")
                            if hasattr(results, 'results'):
                                print(f"Results contain {len(results.results)} items")
                            break  # Only process the first result
                        
                        if results is None:
                            print("No results from model, creating empty results")
                            # Create a dummy result if none was returned
                            class DummyResult:
                                def __init__(self):
                                    self.results = []
                            results = DummyResult()
                            
                    except Exception as e:
                        print(f"ERROR in inference: {e}")
                        # Create a dummy result on error
                        class DummyResult:
                            def __init__(self):
                                self.results = []
                        results = DummyResult()
                
                # Process detections
                detections = process_detections(frame, results)
                
                # Update last inference time
                last_inference_time = current_time
                
                # Handle and visualize detections
                if len(detections) > 0:
                    print(f"Detected {len(detections)} objects")
                    
                    # Save detection frame
                    detection_filename = f"detection_{int(current_time)}.jpg"
                    
                    # Create a copy for drawing
                    annotated_frame = display_frame.copy()
                    
                    # Draw bounding boxes on the frame
                    for detection in detections:
                        x1, y1, x2, y2, score, class_id = detection
                        
                        # Get cat name and color
                        cat_name = CAT_CLASSES.get(class_id, "Unknown")
                        
                        # Print detection details
                        print(f"- {cat_name}: confidence={score:.2f}, box=({x1},{y1},{x2},{y2})")
                        
                        # Draw detection on the frame copy
                        annotated_frame = draw_detection_on_frame(annotated_frame, detection)
                        
                        # Handle detection and relay control
                        handle_detection([x1, y1, x2, y2], frame_width)
                    
                    # Save the annotated frame
                    save_frame(annotated_frame, detection_filename)
                    print(f"Saved annotated detection image to {detection_filename}")
            
            # Save frame periodically (every 5 seconds)
            if current_time - last_frame_save > 5:
                save_frame(frame, f"frame_{int(current_time)}.jpg")
                last_frame_save = current_time
            
            # Calculate and display FPS
            frame_count += 1
            elapsed_time = current_time - start_time
            
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = current_time
                print(f"FPS: {fps:.2f}")
            
            # Short sleep to prevent high CPU usage
            time.sleep(0.01)
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            break
    
    # Clean up
    cleanup_camera(camera)
    if DEV_MODE:
        pygame.mixer.quit()
    print("Program ended successfully")

except Exception as e:
    print(f"Error: {e}")
    if 'camera' in locals():
        cleanup_camera(camera)
    if DEV_MODE:
        pygame.mixer.quit()
    
print("Resources cleaned up")