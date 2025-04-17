#!/usr/bin/env python3
"""
Minimal script to initialize the squirt relay to OFF state and exit immediately.
This helps diagnose if the issue is with the initialization or elsewhere.
"""

import RPi.GPIO as GPIO
import time

# SQUIRT relay pin
SQUIRT_PIN = 5

try:
    # Setup GPIO
    print("Setting up GPIO...")
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    # Initialize pin as output
    GPIO.setup(SQUIRT_PIN, GPIO.OUT)
    
    # Set to HIGH which should turn it OFF
    print(f"Setting pin {SQUIRT_PIN} to HIGH (should be OFF)...")
    GPIO.output(SQUIRT_PIN, GPIO.HIGH)
    
    # Small delay to make sure it takes effect
    time.sleep(0.5)
    
    print("GPIO initialized, squirt relay should be OFF")
    print("Script will exit. Relay should stay OFF after exit.")
finally:
    # Don't run GPIO.cleanup() to see if the pin state persists
    print("Exiting without GPIO cleanup to maintain pin state.") 