#!/bin/bash
# Script to run the tracking test with proper permissions

echo "Running tracking calibration test..."
echo "This will test different relay activation durations to find the optimal time."
echo "The script will detect cats/dogs in the frame and activate relays to center them."
echo ""
echo "DeGirum Requirements:"
echo " - Hailo AI Kit accelerator must be connected via USB"
echo " - DeGirum software must be installed (pip install degirum)"
echo " - DeGirum service must be running (sudo systemctl start degirum)"
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
echo " - Phase 1: 0.2 seconds"
echo " - Phase 2: 0.3 seconds"
echo " - Phase 3: 0.4 seconds"
echo " - Phase 4: 0.5 seconds"
echo ""
echo "Troubleshooting DeGirum issues:"
echo " - Ensure Hailo AI Kit is properly connected and powered via USB"
echo " - Make sure the green light on the Hailo device is on"
echo " - Check DeGirum service status: sudo systemctl status degirum"
echo " - If service isn't running: sudo systemctl start degirum"
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

# Check if DeGirum service is running
if systemctl is-active --quiet degirum; then
    echo "DeGirum service is running"
else
    echo "WARNING: DeGirum service is not running!"
    echo "Starting DeGirum service..."
    sudo systemctl start degirum
    sleep 2
    if systemctl is-active --quiet degirum; then
        echo "DeGirum service started successfully"
    else
        echo "Failed to start DeGirum service. Detection may not work properly."
        echo "Continue anyway? (y/n)"
        read CONTINUE
        if [ "$CONTINUE" != "y" ]; then
            echo "Exiting. Try starting the service manually with: sudo systemctl start degirum"
            exit 1
        fi
    fi
fi

# Make the test script executable
chmod +x test_tracking.py

# Run the test script with sudo if needed (for GPIO access)
if [[ "$EUID" -ne 0 ]]; then
    echo "Running with sudo for GPIO access..."
    sudo python3 test_tracking.py
else
    python3 test_tracking.py
fi

echo ""
echo "Test completed. Check the tracking_tests directory for recorded videos."
echo "Review the videos to determine which activation duration works best for your setup." 