#!/bin/bash

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment
source "$DIR/cat_venv/bin/activate"

# Get the full path to the Python interpreter in the virtual environment
PYTHON_PATH=$(which python3)

# Run the script with sudo, preserving the virtual environment
sudo -E env "PATH=$PATH" "PYTHONPATH=$PYTHONPATH" "$PYTHON_PATH" "$DIR/main.py" 