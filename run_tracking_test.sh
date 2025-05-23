#!/bin/bash
# Script to run the tracking test with proper permissions

echo "Running tracking calibration test..."
echo "This will test different relay activation durations to find the optimal time."
echo "The script will detect cats/dogs in the frame and activate relays to center them."
echo ""
echo "NEW: Simplified 3-Zone Tracking System"
echo " - Left Zone: Moves right for exactly 0.05 seconds"
echo " - Center Zone: No movement (wide ±30% of frame width)"
echo " - Right Zone: Moves left for exactly 0.05 seconds"
echo ""
echo "Key Features:"
echo " - Minimalist approach with only 3 zones"
echo " - Very short 0.05s movements to prevent overshooting"
echo " - Extra-wide center zone (60% of frame width)"
echo " - Longer 0.5s cooldown between movements"
echo " - Limits to 2 consecutive movements in same direction"
echo " - Simplified decision making for more predictable tracking"
echo ""
echo "DeGirum Requirements:"
echo " - Hailo AI Kit accelerator must be connected via USB"
echo " - DeGirum software must be installed (pip install degirum)"
echo " - Model zoo should be available at /home/pi5/degirum_model_zoo"
echo ""
echo "The test phases no longer affect movement duration (fixed at 0.05s)."
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
echo "Review the videos to determine which tracking approach works best for your setup." 