#!/bin/bash
# Script to run the tracking test with proper permissions

echo "Running tracking calibration test..."
echo "This will test different relay activation durations to find the optimal time."
echo "The script will detect cats/dogs in the frame and activate relays to center them."
echo ""
echo "Fixed issues:"
echo " - Corrected model loading to use yolo11s model (same as main.py)"
echo " - Fixed image preprocessing for the DeGirum model"
echo " - Improved inference method to match main.py's approach"
echo " - Enhanced detection output processing with detailed diagnostics"
echo " - Added relay testing at startup to verify hardware connections"
echo " - Improved multiple detection format handling (works with different model outputs)"
echo " - Added extensive debug output to help diagnose issues"
echo ""
echo "The test will run through 4 phases with different activation durations:"
echo " - Phase 1: 0.2 seconds"
echo " - Phase 2: 0.3 seconds"
echo " - Phase 3: 0.4 seconds"
echo " - Phase 4: 0.5 seconds"
echo ""
echo "Troubleshooting tips:"
echo " - If no detections appear, ensure a cat or dog is clearly visible"
echo " - Try adjusting the lighting conditions for better detection"
echo " - Check relay connections if activations don't seem to work"
echo " - Set TEST_SQUIRT=True in the code to test squirt activation"
echo ""
echo "Press Ctrl+C to stop the test at any time."
echo ""

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