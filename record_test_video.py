#!/usr/bin/env python3
"""
Simple script to record a 60-second test video using the Raspberry Pi camera.
Optimized for object detection performance while maintaining good quality.
"""

import time
import os
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, Quality
from picamera2.outputs import FfmpegOutput

# --- Configuration ---
VIDEO_DURATION = 60  # Duration in seconds
# Use 640x640 resolution to match YOLO model input size
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 640
VIDEO_FPS = 30
OUTPUT_DIR = "test_videos"  # Directory to save videos
OUTPUT_FILENAME = f"pi_camera_test_{VIDEO_WIDTH}x{VIDEO_HEIGHT}_{VIDEO_DURATION}s.mp4"

# Camera settings optimized for detection
CAMERA_SETTINGS = {
    "AeEnable": True,  # Auto exposure
    "AwbEnable": True,  # Auto white balance
    "AeExposureMode": 0,  # Normal exposure mode
    "AeMeteringMode": 0,  # Center-weighted metering
    "ExposureTime": 0,  # Let auto-exposure handle it
    "AnalogueGain": 1.0,  # Let auto-gain handle it
    "ColourGains": (1.0, 1.0),  # Let auto-white balance handle it
    "Brightness": 0.0,  # Default brightness
    "Contrast": 1.0,  # Default contrast
    "Saturation": 1.0,  # Default saturation
    "FrameRate": VIDEO_FPS  # Set desired frame rate
}

def record_test_video(output_path, duration):
    """Initializes camera, records video for specified duration, and saves it."""
    print("Initializing camera...")
    picam2 = Picamera2()

    try:
        # Configure video with detection-optimized settings
        video_config = picam2.create_video_configuration(
            main={
                "size": (VIDEO_WIDTH, VIDEO_HEIGHT),
                "format": "XBGR8888"  # Use XBGR for better color handling
            },
            controls=CAMERA_SETTINGS
        )
        
        # Apply the configuration
        picam2.configure(video_config)
        print(f"Configured for {VIDEO_WIDTH}x{VIDEO_HEIGHT} @ {VIDEO_FPS} FPS")
        print("Camera settings:")
        for setting, value in CAMERA_SETTINGS.items():
            print(f"  {setting}: {value}")

        # Setup encoder with balanced quality settings
        encoder = H264Encoder(
            bitrate=2000000  # 2 Mbps is sufficient for 640x640
        )
        
        # Setup output with efficient codec settings
        output = FfmpegOutput(
            output_path,
            audio=False,  # No audio needed
            vcodec='libx264',  # Use x264 codec
            preset='faster',  # Faster encoding for lower latency
            tune='zerolatency'  # Optimize for low latency
        )

        print(f"\nStarting recording to {output_path} for {duration} seconds...")
        print("Press Ctrl+C to stop recording early.")

        # Start recording
        picam2.start_recording(encoder, output)

        # Wait for the specified duration
        start_time = time.time()
        while time.time() - start_time < duration:
            time.sleep(0.1)  # Check more frequently for smoother interruption
            elapsed = time.time() - start_time
            if elapsed % 5 == 0:  # Print progress every 5 seconds
                print(f"Recording... {elapsed:.1f}/{duration} seconds")

        # Stop recording
        picam2.stop_recording()
        print("\nRecording completed successfully.")

    except KeyboardInterrupt:
        print("\nRecording stopped early by user.")
        picam2.stop_recording()
    except Exception as e:
        print(f"\nAn error occurred during recording: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close the camera connection
        picam2.close()
        print("Camera closed.")
        
        # Verify the output file
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
            print(f"Video saved successfully to: {output_path}")
            print(f"File size: {file_size:.1f} MB")
        else:
            print(f"Failed to save video to {output_path}")


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
            print(f"Created output directory: {OUTPUT_DIR}")
        except OSError as e:
            print(f"Error creating directory {OUTPUT_DIR}: {e}")
            # Fallback to current directory if unable to create
            OUTPUT_DIR = "."

    # Define the full output path
    full_output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    # Start recording
    record_test_video(full_output_path, VIDEO_DURATION)