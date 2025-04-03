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
                
                # DEBUG: Print out the type and structure of the first result
                first_result = results.results[0]
                print(f"First result type: {type(first_result)}")
                print(f"First result attributes: {dir(first_result)}")
                
                # Try several different ways to access the detection data
                for detection in results.results:
                    try:
                        # Method 1: Direct access if detection is a dictionary
                        if isinstance(detection, dict):
                            if 'bbox' in detection:
                                bbox = detection['bbox']
                                x1, y1, x2, y2 = map(int, bbox)
                                score = float(detection.get('score', 0.0))
                                class_id = int(detection.get('category_id', 0))
                                
                                cat_name = CAT_CLASSES.get(class_id, f"Unknown class {class_id}")
                                print(f"Method 1 - Detection: {cat_name}, bbox={x1},{y1},{x2},{y2}, score={score:.2f}")
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
                                print(f"Unknown bbox format: {detection.bbox}")
                                continue
                                
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
                                
                            cat_name = CAT_CLASSES.get(class_id, f"Unknown class {class_id}")
                            print(f"Method 2 - Detection: {cat_name}, bbox={x1},{y1},{x2},{y2}, score={score:.2f}")
                            detections.append((x1, y1, x2, y2, score, class_id))
                            continue
                        
                        # Method 3: Try to access via image_overlay if available
                        if hasattr(detection, 'image_overlay'):
                            print("Found image_overlay attribute, but can't extract coordinates directly")
                            continue
                            
                        # If we got here, print the detection object to see what it contains
                        print(f"Couldn't process detection: {detection}")
                        if hasattr(detection, '__dict__'):
                            print(f"Detection __dict__: {detection.__dict__}")
                            
                    except Exception as e:
                        print(f"Error processing detection: {e}")
                        import traceback
                        traceback.print_exc()
                
            # If we still have no detections, try alternate approaches        
            elif hasattr(results, 'bboxes') and hasattr(results, 'scores'):
                print(f"Using bboxes/scores attributes")
                for i in range(len(results.bboxes)):
                    bbox = results.bboxes[i]
                    score = results.scores[i]
                    class_id = results.class_ids[i] if hasattr(results, 'class_ids') else 0
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    cat_name = CAT_CLASSES.get(class_id, f"Unknown class {class_id}")
                    print(f"Alt method - Detection: {cat_name}, bbox={x1},{y1},{x2},{y2}, score={score:.2f}")
                    detections.append((x1, y1, x2, y2, score, class_id))
                    
            # Try direct access to the image overlay if available
            elif hasattr(results, 'image_overlay') and results.image_overlay is not None:
                print("Using image_overlay, but can't extract coordinates for detections")
                
            # Last resort - dump everything about the results object
            if len(detections) == 0:
                print(f"No detections found. Results dump:")
                for attr in dir(results):
                    if not attr.startswith('__'):
                        try:
                            value = getattr(results, attr)
                            print(f"  {attr}: {value}")
                        except:
                            print(f"  {attr}: <error accessing>")
        
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
        
        # Get cat name and color
        cat_name = CAT_CLASSES.get(class_id, f"Unknown-{class_id}")
        color = COLORS.get(cat_name.lower(), COLORS['unknown'])
        
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
        label_text = f"{cat_name}: {score:.2f}"
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
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLORS['white'], 2)
        
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
                            
                            # Print more information about the results to understand their structure
                            if hasattr(results, 'results'):
                                print(f"Results contain {len(results.results)} items")
                                
                                # Try to extract the result information using the image_overlay
                                if hasattr(results, 'image_overlay') and results.image_overlay is not None:
                                    print("Image overlay is available - using for visual verification")
                                    # Save a copy of the overlay image for debugging
                                    overlay_path = f"overlay_{int(current_time)}.jpg"
                                    cv2.imwrite(overlay_path, results.image_overlay)
                                    print(f"Saved raw overlay image to {overlay_path}")
                                
                                # Print detailed information about the first detection
                                if len(results.results) > 0:
                                    first_det = results.results[0]
                                    print(f"First detection type: {type(first_det)}")
                                    
                                    # If it's a dictionary, print its contents
                                    if isinstance(first_det, dict):
                                        for k, v in first_det.items():
                                            print(f"  {k}: {v}")
                                    # If it's an object, print its attributes
                                    elif hasattr(first_det, '__dict__'):
                                        for k, v in first_det.__dict__.items():
                                            print(f"  {k}: {v}")
                                    # Otherwise try to list its attributes
                                    else:
                                        for attr in dir(first_det):
                                            if not attr.startswith('__'):
                                                try:
                                                    val = getattr(first_det, attr)
                                                    print(f"  {attr}: {val}")
                                                except:
                                                    print(f"  {attr}: <error accessing>")
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
                        import traceback
                        traceback.print_exc()
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
                        relative_position = handle_detection([x1, y1, x2, y2], frame_width)
                        
                        # Draw directional indicators
                        height, width = annotated_frame.shape[:2]
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
                    
                    # Save the annotated frame
                    save_frame(annotated_frame, detection_filename)
                    print(f"Saved annotated detection image to {detection_filename}")
            
            # Save frame periodically (every 5 seconds)
            if current_time - last_frame_save > 5:
                # Create a debug frame with guidelines
                debug_frame = frame.copy()
                height, width = debug_frame.shape[:2]
                center_x = width // 2
                
                # Draw center line
                cv2.line(debug_frame, (center_x, 0), (center_x, height), (0, 255, 255), 2)
                
                # Draw threshold lines
                left_threshold = int(width * (0.5 - CENTER_THRESHOLD))
                right_threshold = int(width * (0.5 + CENTER_THRESHOLD))
                cv2.line(debug_frame, (left_threshold, 0), (left_threshold, height), (0, 0, 255), 2)
                cv2.line(debug_frame, (right_threshold, 0), (right_threshold, height), (0, 0, 255), 2)
                
                # Add timestamp and FPS
                time_text = f"Time: {time.strftime('%H:%M:%S')}"
                fps_text = f"FPS: {fps:.1f}"
                cv2.rectangle(debug_frame, (0, 0), (250, 70), (0, 0, 0), -1)
                cv2.putText(debug_frame, time_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(debug_frame, fps_text, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Save the debug frame
                save_frame(debug_frame, f"debug_{int(current_time)}.jpg")
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