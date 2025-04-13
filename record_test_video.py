#!/usr/bin/env python3
"""
Script to record a test video using the Raspberry Pi camera.
Records a 2-minute video at 640x640 resolution, optimized for the YOLO11s model.
"""

import sys
import time
import os
import subprocess
import site

def check_and_install_packages():
    """Check for required packages and install them if missing."""
    # System packages
    system_packages = [
        "python3-picamera2",
        "libcamera-tools",
        "ffmpeg",
        "python3-libcamera"  # Add the Python bindings for libcamera
    ]
    
    print("Checking for required system packages...")
    missing_system_packages = []
    
    for package in system_packages:
        try:
            subprocess.run(["dpkg", "-l", package], check=True, 
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            missing_system_packages.append(package)
    
    if missing_system_packages:
        print(f"Missing required system packages: {', '.join(missing_system_packages)}")
        print("Installing missing system packages...")
        try:
            subprocess.run(["sudo", "apt-get", "update"], check=True)
            subprocess.run(["sudo", "apt-get", "install", "-y"] + missing_system_packages, check=True)
            print("System packages installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error installing system packages: {e}")
            print("Please run the following commands manually:")
            print("sudo apt-get update")
            print(f"sudo apt-get install -y {' '.join(missing_system_packages)}")
            sys.exit(1)
    else:
        print("All required system packages are installed!")
    
    # Add system site-packages to Python path
    system_site_packages = [p for p in site.getsitepackages() if 'dist-packages' in p]
    if system_site_packages:
        sys.path.extend(system_site_packages)
        print(f"Added system site-packages to Python path: {system_site_packages[0]}")
    
    print("\nChecking for required Python packages...")
    try:
        # Try to import libcamera to check if it's accessible
        import libcamera
        print("libcamera is accessible!")
    except ImportError:
        print("Error: libcamera is not accessible in the virtual environment.")
        print("Please run the following command to make system packages available:")
        print("python3 -m venv --system-site-packages cat_venv")
        print("Then reactivate your virtual environment:")
        print("source cat_venv/bin/activate")
        sys.exit(1)

# Check packages before importing picamera2
check_and_install_packages()

# Now import picamera2 after ensuring packages are installed
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
    
    # Configure video settings with proper color format
    video_config = picam2.create_video_configuration(
        main={"size": (VIDEO_WIDTH, VIDEO_HEIGHT), "format": "YUV420"},  # Changed to YUV420
        controls={
            "FrameRate": VIDEO_FPS,
            "AwbMode": 0,  # Auto white balance
            "AeEnable": True,   # Auto exposure
            "ExposureTime": 10000,  # Initial exposure time in microseconds
            "AnalogueGain": 1.0,    # Initial analog gain
            "ColourGains": (1.0, 1.0),  # Initial color gains
            "Brightness": 0.0,  # Adjust brightness
            "Contrast": 1.0,    # Adjust contrast
            "Saturation": 1.0   # Adjust saturation
        }
    )
    
    # Apply configuration
    picam2.configure(video_config)
    
    # Set additional controls for better quality
    picam2.set_controls({
        "AwbEnable": True,  # Auto white balance
        "AeEnable": True,   # Auto exposure
        "ExposureTime": 10000,  # Initial exposure time in microseconds
        "AnalogueGain": 1.0,    # Initial analog gain
        "ColourGains": (1.0, 1.0),  # Initial color gains
        "Brightness": 0.0,  # Adjust brightness
        "Contrast": 1.0,    # Adjust contrast
        "Saturation": 1.0   # Adjust saturation
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
        # Configure FFmpeg output with correct parameters
        encoder = H264Encoder(bitrate=5000000)  # 5 Mbps
        output = FfmpegOutput(
            output_path,
            audio=False,
            video=True,
            format='mp4',
            codec='h264',
            framerate=30,
            quality=23
        )
        
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