#!/bin/bash
# Script to run the tracking test with proper permissions

echo "Running tracking calibration test..."
echo "This will test different relay activation durations to find the optimal time."
echo "The script will detect cats/dogs in the frame and activate relays to center them."
echo ""
echo "NEW: Improved Tracking System"
echo " - Position-based tracking with numerical position scale (-5 to +5)"
echo " - Smoother movements with acceleration/deceleration"
echo " - Longer center reset time (60 seconds vs 10 seconds)"
echo " - Proportional movement duration based on distance from center"
echo " - Hysteresis and filtering to avoid jittery movement"
echo " - Gradual center return instead of abrupt reset"
echo ""
echo "DeGirum Requirements:"
echo " - Hailo AI Kit accelerator must be connected via USB"
echo " - DeGirum software must be installed (pip install degirum)"
echo " - Model zoo should be available at /home/pi5/degirum_model_zoo"
echo ""
echo "Fixed issues:"
echo " - Corrected model loading to use yolo11s model (same as main.py)"
echo " - Fixed image preprocessing for the DeGirum model"
echo " - Matched main.py exactly: now using model.predict_batch([frame]) for inference"
echo " - Aligned detection processing with main.py's implementation"
echo " - Added frame skipping (every 3rd frame) to match main.py's approach"
echo " - Enhanced detection output processing with detailed diagnostics"
echo " - Added relay testing at startup to verify hardware connections"
echo " - Improved multiple detection format handling (works with different model outputs)"
echo " - Added extensive debug output to help diagnose issues"
echo " - Enhanced DeGirum detection and initialization process"
echo ""
echo "The test will run through 4 phases with different activation durations:"
echo " - Phase 1: 0.2 seconds - Base duration, scaled by distance from center"
echo " - Phase 2: 0.3 seconds - Base duration, scaled by distance from center"
echo " - Phase 3: 0.4 seconds - Base duration, scaled by distance from center"
echo " - Phase 4: 0.5 seconds - Base duration, scaled by distance from center"
echo ""
echo "Troubleshooting DeGirum issues:"
echo " - Ensure Hailo AI Kit is properly connected and powered via USB"
echo " - Make sure the green light on the Hailo device is on"
echo " - Check that the Hailo device is recognized: lsusb | grep Hailo"
echo " - If detection still fails, try rebooting: sudo reboot"
echo ""
echo "Troubleshooting detection issues:"
echo " - If no detections appear, ensure a cat or dog is clearly visible"
echo " - Try adjusting the lighting conditions for better detection"
echo " - Check relay connections if activations don't seem to work"
echo " - Set TEST_SQUIRT=True in the code to test squirt activation"
echo ""
echo "Press Ctrl+C to stop the test at any time."
echo ""

# Check if Hailo device is connected
if lsusb | grep -q Hailo; then
    echo "Hailo device detected"
else
    echo "WARNING: Hailo device not detected!"
    echo "Please ensure the Hailo AI Kit is properly connected via USB."
    echo "Continue anyway? (y/n)"
    read CONTINUE
    if [ "$CONTINUE" != "y" ]; then
        echo "Exiting. Connect the Hailo device and try again."
        exit 1
    fi
fi

# Make the test script executable
chmod +x test_tracking.py

# Detect if we're already in a virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Running in virtual environment: $VIRTUAL_ENV"
    PYTHON_PATH="$VIRTUAL_ENV/bin/python3"
else
    echo "No virtual environment detected, using system Python"
    PYTHON_PATH="python3"
    
    # Check if cat_venv exists and use it if available
    if [ -d "$HOME/Projects/hailo_cat_detector/cat_venv" ]; then
        echo "Found cat_venv, using it instead"
        PYTHON_PATH="$HOME/Projects/hailo_cat_detector/cat_venv/bin/python3"
    fi
fi

# Run the test script with sudo if needed (for GPIO access)
if [[ "$EUID" -ne 0 ]]; then
    echo "Running with sudo to get GPIO access, preserving Python environment..."
    # Use sudo with -E to preserve environment variables, and specify the exact Python executable
    sudo -E $PYTHON_PATH test_tracking.py
else
    $PYTHON_PATH test_tracking.py
fi

echo ""
echo "Test completed. Check the tracking_tests directory for recorded videos."
echo "Review the videos to determine which activation duration works best for your setup." 