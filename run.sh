#!/bin/bash
set -e  # Exit on any error

echo "======================================================"
echo "     Hailo Cat Detector System - Startup Script       "
echo "======================================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Warning: Not running as root (sudo). Some hardware features may be limited."
  echo "Consider running with 'sudo ./run.sh' for full hardware access."
  echo "Continuing anyway..."
  echo ""
fi

# Create required directories
echo "Setting up required directories..."
mkdir -p logs
chmod 777 logs
mkdir -p recordings
chmod 777 recordings

# Check for Python and required packages
echo "Checking for Python and required packages..."
if ! command -v python3 &> /dev/null; then
  echo "ERROR: Python 3 is not installed or not in PATH"
  exit 1
fi

# Check for DeGirum package
python3 -c "import degirum" &> /dev/null
if [ $? -ne 0 ]; then
  echo "ERROR: DeGirum Python package not found. Please install with:"
  echo "pip install degirum"
  exit 1
fi

# Check for OpenCV
python3 -c "import cv2" &> /dev/null
if [ $? -ne 0 ]; then
  echo "ERROR: OpenCV Python package not found. Please install with:"
  echo "pip install opencv-python"
  exit 1
fi

# Check for RPi.GPIO
python3 -c "import RPi.GPIO" &> /dev/null
if [ $? -ne 0 ]; then
  echo "ERROR: RPi.GPIO Python package not found. Please install with:"
  echo "pip install RPi.GPIO"
  exit 1
fi

# Check for Hailo device
echo "Checking for Hailo device..."
if command -v hailortcli &> /dev/null; then
  echo "Running Hailo device check..."
  HAILO_OUTPUT=$(hailortcli device show 2>&1)
  HAILO_STATUS=$?
  
  if [ $HAILO_STATUS -eq 0 ]; then
    echo "Hailo device detected:"
    echo "$HAILO_OUTPUT" | grep -E "Device|Platform|Type|Power" | sed 's/^/  /'
    echo "Hailo device is ready for inference."
  else
    echo "WARNING: Hailo device check failed with status: $HAILO_STATUS"
    echo "Error output: $HAILO_OUTPUT"
    echo "This might be due to permissions or hardware issues."
    echo "Continuing anyway, but acceleration might not work..."
  fi
else
  echo "WARNING: hailortcli not found in PATH. Is the Hailo SDK properly installed?"
  echo "Continuing anyway, but hardware acceleration might not work..."
fi

echo ""
echo "Initializing GPIO pins to safe state..."
python3 init_gpio_low.py || echo "WARNING: GPIO initialization failed. Check permissions."

echo ""
echo "Starting main cat detection program..."
echo "------------------------------------------------------"
python3 main.py "$@"

EXIT_CODE=$?
echo "------------------------------------------------------"
if [ $EXIT_CODE -eq 0 ]; then
  echo "Program exited successfully."
else
  echo "Program exited with error code: $EXIT_CODE"
fi 