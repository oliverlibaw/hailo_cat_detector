# This project detects cats and triggers a water gun to squirt them. It runs on a Raspberry Pi 5 with a Hailo AI Kit and a four-relay HAT.

import os
import cv2
import time
import numpy as np
import random
import torch
from ultralytics import YOLO
import pygame

# Development mode flag - set to True when developing on MacBook
DEV_MODE = False  # Set to False for Raspberry Pi deployment

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
CONFIDENCE_THRESHOLD = 0.25  # Confidence Threshold
MODEL_INPUT_SIZE = (640, 640)  # YOLOv8n input size
CENTER_THRESHOLD = 0.1  # Threshold for determining if object is left/right of center
RELAY_CENTER_DURATION = 0.2  # Duration to activate center relay
RELAY_CENTER_COOLDOWN = 1.0  # Cooldown period for center relay

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
            return frame
        except Exception as e:
            print(f"Failed to capture frame from Pi camera: {e}")
            raise

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
    """Load the AI model with proper error handling"""
    try:
        if DEV_MODE:
            print("Development mode: Loading custom YOLOv8n model for cat detection...")
            model_path = 'yolov8n_cats.pt'  # Your custom model
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model = YOLO(model_path)
            print("Successfully loaded custom cat detection model")
        else:
            print("Loading Degirum model for Hailo accelerator...")
            import degirum as dg
            
            # Connect to the Hailo accelerator
            zoo = dg.connect_model_zoo(inference_host_address, zoo_url, token)
            model = zoo.load_model(model_name)
            print("Successfully loaded Degirum model")
        
        # Print model information
        if DEV_MODE:
            print(f"Model loaded: {model.__class__.__name__}")
            print(f"Model size: {os.path.getsize('yolov8n_cats.pt') / (1024*1024):.2f} MB")
            print("Detecting cats: Gary, George, and Fred")
        else:
            print(f"Model loaded: {type(model).__name__}")
            print(f"Model name: {model_name}")
            print("Running on Hailo accelerator")
        
        return model
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        print("Detailed error traceback:")
        import traceback
        traceback.print_exc()
        raise

try:
    # Load AI model with proper error handling
    model = load_model()
    setup_sound()
    
    # Setup camera
    camera = setup_camera()
    
    # Print model info
    print(f"Model: {model_name}")
    print(f"Model input size: {MODEL_INPUT_SIZE}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Development mode: {'Enabled' if DEV_MODE else 'Disabled'}")
    
    start_time = time.time()
    frame_count = 0
    fps = 0
    
    while True:
        frame = get_frame(camera)
        frame_width = frame.shape[1]
        
        # Create display frame and add overlay
        display_frame = frame.copy()
        draw_overlay(display_frame)
        
        # Perform inference
        results = model(frame, conf=CONFIDENCE_THRESHOLD)[0]
        
        if len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                score = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Get cat name and color
                cat_name = CAT_CLASSES.get(class_id, "Unknown")
                color = COLORS.get(cat_name.lower(), COLORS['unknown'])
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add label and confidence
                label_text = f"{cat_name}: {score:.2f}"
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                cv2.rectangle(display_frame, 
                            (x1, y1 - text_size[1] - 5), 
                            (x1 + text_size[0], y1), 
                            COLORS['black'], -1)
                
                cv2.putText(display_frame, label_text, 
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['white'], 2)
                
                # Handle detection and relay control
                relative_position = handle_detection([x1, y1, x2, y2], frame_width)
                draw_overlay(display_frame, relative_position)
        
        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Add FPS text
        fps_text = f"FPS: {fps:.2f}"
        cv2.rectangle(display_frame, (5, 5), (120, 35), COLORS['black'], -1)
        cv2.putText(display_frame, fps_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['green'], 2)
        
        # Display the frame
        cv2.imshow("Object Detection", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    # Clean up
    cleanup_camera(camera)
    cv2.destroyAllWindows()
    if DEV_MODE:
        pygame.mixer.quit()
    print("Program ended successfully")

except KeyboardInterrupt:
    print("Program terminated by user")
    if 'camera' in locals():
        cleanup_camera(camera)
    cv2.destroyAllWindows()
    if DEV_MODE:
        pygame.mixer.quit()
except Exception as e:
    print(f"Error: {e}")
    if 'camera' in locals():
        cleanup_camera(camera)
    cv2.destroyAllWindows()
    if DEV_MODE:
        pygame.mixer.quit()
    
print("Resources cleaned up")