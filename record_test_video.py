#!/usr/bin/env python3
"""
Simple script to record a 60-second test video using the Raspberry Pi camera.
"""

import time
import os
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, Quality
from picamera2.outputs import FfmpegOutput

# --- Configuration ---
VIDEO_DURATION = 60  # Duration in seconds
# Consider using a resolution native to your camera for potentially better quality,
# e.g., (1920, 1080), and let the detection script handle letterboxing.
# Keeping 640x640 as per the previous script for consistency.
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 640
VIDEO_FPS = 30
OUTPUT_DIR = "test_videos"  # Directory to save videos
# You can customize the filename, e.g., add a timestamp
# OUTPUT_FILENAME = f"pi_test_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
OUTPUT_FILENAME = f"pi_camera_test_{VIDEO_WIDTH}x{VIDEO_HEIGHT}_{VIDEO_DURATION}s.mp4"
# --- End Configuration ---

def record_test_video(output_path, duration):
    """Initializes camera, records video for specified duration, and saves it."""
    print("Initializing camera...")
    picam2 = Picamera2()

    # Simplified video configuration - relying more on defaults
    video_config = picam2.create_video_configuration(
        main={"size": (VIDEO_WIDTH, VIDEO_HEIGHT), "format": "XBGR8888"},
        controls={"FrameRate": VIDEO_FPS} # Set desired frame rate
    )
    picam2.configure(video_config)
    print(f"Configured for {VIDEO_WIDTH}x{VIDEO_HEIGHT} @ {VIDEO_FPS} FPS.")

    # Setup encoder and output
    # Using quality-based encoding is generally preferred over fixed bitrate
    encoder = H264Encoder(quality=Quality.HIGH) # Options: VERY_LOW, LOW, MEDIUM, HIGH, VERY_HIGH
    output = FfmpegOutput(output_path) # Saves to MP4 using ffmpeg

    print(f"Starting recording to {output_path} for {duration} seconds...")
    print("Press Ctrl+C to stop recording early.")

    try:
        # Start recording using the simplified method
        picam2.start_recording(encoder, output)

        # Wait for the specified duration while keeping the script alive
        # (start_recording runs in a background thread)
        time.sleep(duration)

        # Stop recording
        picam2.stop_recording()
        print("Recording stopped.")

    except KeyboardInterrupt:
        print("\nRecording stopped early by user.")
        # Ensure recording stops if interrupted
        picam2.stop_recording()
    except Exception as e:
        print(f"An error occurred during recording: {e}")
    finally:
        # Close the camera connection
        picam2.close()
        print("Camera closed.")
        if os.path.exists(output_path):
             print(f"Video saved successfully to: {output_path}")
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