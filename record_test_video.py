#!/usr/bin/env python3
"""
Script to record a test video using the Raspberry Pi camera.
Records a 2-minute video at 640x640 resolution, optimized for the YOLO11s model.
"""

import time
import os
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput

# Configuration
VIDEO_DURATION = 120  # 2 minutes in seconds
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 640
VIDEO_FPS = 30
OUTPUT_DIR = "test_videos"
OUTPUT_FILENAME = "pi_camera_test_640x640.mp4"

def setup_camera():
    """Initialize and configure the PiCamera2."""
    print("Initializing camera...")
    picam2 = Picamera2()
    
    # Configure video settings
    video_config = picam2.create_video_configuration(
        main={"size": (VIDEO_WIDTH, VIDEO_HEIGHT), "format": "RGB888"},
        controls={"FrameRate": VIDEO_FPS}
    )
    
    # Apply configuration
    picam2.configure(video_config)
    
    # Set additional controls for better quality
    picam2.set_controls({
        "AwbEnable": True,  # Auto white balance
        "AeEnable": True,   # Auto exposure
        "ExposureTime": 10000,  # Initial exposure time in microseconds
        "AnalogueGain": 1.0,    # Initial analog gain
        "ColourGains": (1.0, 1.0)  # Initial color gains
    })
    
    return picam2

def record_video():
    """Record a test video with the configured settings."""
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    # Initialize camera
    picam2 = setup_camera()
    
    try:
        # Create encoder and output
        encoder = H264Encoder()
        output = FfmpegOutput(output_path)
        
        # Start recording
        print(f"Starting recording for {VIDEO_DURATION} seconds...")
        print(f"Output will be saved to: {output_path}")
        print("Press Ctrl+C to stop recording early")
        
        picam2.start_recording(encoder, output)
        start_time = time.time()
        
        # Record for specified duration
        while time.time() - start_time < VIDEO_DURATION:
            time.sleep(0.1)  # Small sleep to prevent CPU overload
            elapsed = time.time() - start_time
            if elapsed % 5 == 0:  # Print progress every 5 seconds
                print(f"Recording... {elapsed:.1f}/{VIDEO_DURATION} seconds")
        
        # Stop recording
        print("Recording complete!")
        picam2.stop_recording()
        
    except KeyboardInterrupt:
        print("\nRecording stopped by user")
        picam2.stop_recording()
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        picam2.close()
    
    print(f"Video saved to: {output_path}")
    print(f"Video properties: {VIDEO_WIDTH}x{VIDEO_HEIGHT} @ {VIDEO_FPS} FPS")

if __name__ == "__main__":
    record_video() 