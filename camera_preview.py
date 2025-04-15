#!/usr/bin/env python3
"""
Script to preview the Raspberry Pi camera with adjustable settings.
This allows testing camera settings before recording a video.
"""

import time
import os
import sys
from picamera2 import Picamera2
import cv2

# Default settings (starting point)
DEFAULT_SETTINGS = {
    "AeEnable": True,           # Auto exposure
    "AwbEnable": True,          # Auto white balance
    "AnalogueGain": 1.5,        # Gain value
    "Brightness": 0.2,          # Brightness value
    "Contrast": 1.1,            # Contrast value
    "Saturation": 1.1,          # Saturation value
    "ExposureValue": 0.5        # EV compensation
}

def preview_camera():
    """Preview camera with adjustable settings via keyboard input."""
    print("Initializing camera preview...")
    picam2 = Picamera2()
    
    # Current settings (will be modified during preview)
    current_settings = DEFAULT_SETTINGS.copy()
    
    try:
        # Configure video with current settings
        preview_config = picam2.create_preview_configuration(
            main={"size": (640, 640)},
            controls=current_settings
        )
        
        picam2.configure(preview_config)
        picam2.start()
        
        print("\nCamera Preview Controls:")
        print("B/b: Increase/decrease brightness")
        print("C/c: Increase/decrease contrast")
        print("G/g: Increase/decrease gain")
        print("S/s: Increase/decrease saturation")
        print("E/e: Increase/decrease exposure compensation")
        print("R: Reset to default settings")
        print("Q: Quit preview")
        print("P: Print current settings")
        
        # Helper function to apply settings
        def apply_settings():
            picam2.set_controls(current_settings)
            # Also apply noise reduction
            picam2.set_controls({"NoiseReductionMode": 2})
        
        while True:
            # Get the latest frame
            frame = picam2.capture_array()
            
            # Convert from XBGR to BGR for OpenCV display
            if frame.shape[2] == 4:  # If it has 4 channels (XBGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # Display camera preview
            cv2.imshow("Camera Preview", frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            
            # Process key actions
            if key == ord('q'):
                break
            elif key == ord('B'):
                current_settings["Brightness"] = min(1.0, current_settings["Brightness"] + 0.1)
                apply_settings()
                print(f"Brightness: {current_settings['Brightness']:.1f}")
            elif key == ord('b'):
                current_settings["Brightness"] = max(-1.0, current_settings["Brightness"] - 0.1)
                apply_settings()
                print(f"Brightness: {current_settings['Brightness']:.1f}")
            elif key == ord('C'):
                current_settings["Contrast"] = min(2.0, current_settings["Contrast"] + 0.1)
                apply_settings()
                print(f"Contrast: {current_settings['Contrast']:.1f}")
            elif key == ord('c'):
                current_settings["Contrast"] = max(0.0, current_settings["Contrast"] - 0.1)
                apply_settings()
                print(f"Contrast: {current_settings['Contrast']:.1f}")
            elif key == ord('G'):
                current_settings["AnalogueGain"] = min(8.0, current_settings["AnalogueGain"] + 0.25)
                apply_settings()
                print(f"Gain: {current_settings['AnalogueGain']:.2f}")
            elif key == ord('g'):
                current_settings["AnalogueGain"] = max(1.0, current_settings["AnalogueGain"] - 0.25)
                apply_settings()
                print(f"Gain: {current_settings['AnalogueGain']:.2f}")
            elif key == ord('S'):
                current_settings["Saturation"] = min(2.0, current_settings["Saturation"] + 0.1)
                apply_settings()
                print(f"Saturation: {current_settings['Saturation']:.1f}")
            elif key == ord('s'):
                current_settings["Saturation"] = max(0.0, current_settings["Saturation"] - 0.1)
                apply_settings()
                print(f"Saturation: {current_settings['Saturation']:.1f}")
            elif key == ord('E'):
                current_settings["ExposureValue"] = min(4.0, current_settings["ExposureValue"] + 0.5)
                apply_settings()
                print(f"EV: {current_settings['ExposureValue']:.1f}")
            elif key == ord('e'):
                current_settings["ExposureValue"] = max(-4.0, current_settings["ExposureValue"] - 0.5)
                apply_settings()
                print(f"EV: {current_settings['ExposureValue']:.1f}")
            elif key == ord('r'):
                current_settings = DEFAULT_SETTINGS.copy()
                apply_settings()
                print("Reset to default settings")
            elif key == ord('p'):
                print("\nCurrent Settings:")
                for setting, value in current_settings.items():
                    print(f"  {setting}: {value}")
            
    except KeyboardInterrupt:
        print("\nPreview stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        try:
            picam2.stop()
            picam2.close()
        except:
            pass
        cv2.destroyAllWindows()
        
        print("\nFinal Camera Settings:")
        for setting, value in current_settings.items():
            print(f"  {setting}: {value}")
        
        print("\nCopy these settings to your record_test_video.py file if they work well.")

if __name__ == "__main__":
    preview_camera() 