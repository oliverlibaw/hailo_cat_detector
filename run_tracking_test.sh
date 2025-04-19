#!/bin/bash
# Script to run the tracking test with proper permissions

echo "Running tracking calibration test..."
echo "This will test different relay activation durations to find the optimal time."
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