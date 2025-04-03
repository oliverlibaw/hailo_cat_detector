#!/bin/bash

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment
source "$DIR/cat_venv/bin/activate"

# Get the full path to the Python interpreter in the virtual environment
PYTHON_PATH=$(which python3)

# Add system Python packages to PYTHONPATH
SYSTEM_PYTHON_PATH="/usr/lib/python3.11:/usr/lib/python3/dist-packages"
export PYTHONPATH="$SYSTEM_PYTHON_PATH:$PYTHONPATH"

# Run the script with sudo, preserving the virtual environment
sudo -E env "PATH=$PATH" "PYTHONPATH=$PYTHONPATH" "$PYTHON_PATH" "$DIR/main.py" 