#!/bin/bash

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment
source "$DIR/cat_venv/bin/activate"

# Get the full path to the Python interpreter in the virtual environment
PYTHON_PATH=$(which python3)

# Get the virtual environment's site-packages directory
VENV_SITE_PACKAGES="$DIR/cat_venv/lib/python3.11/site-packages"

# Add virtual environment packages first, then system packages
export PYTHONPATH="$VENV_SITE_PACKAGES:/usr/lib/python3.11:/usr/lib/python3/dist-packages"

# Run the script with sudo, preserving the virtual environment
sudo -E env "PATH=$PATH" "PYTHONPATH=$PYTHONPATH" "$PYTHON_PATH" "$DIR/main.py" 