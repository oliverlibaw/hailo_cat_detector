#!/usr/bin/env python3
"""
Minimal script to initialize the squirt relay to LOW state and exit immediately.
This tests if LOW is actually the OFF state for this relay.
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
    
    # Set to LOW to see if that turns it OFF
    print(f"Setting pin {SQUIRT_PIN} to LOW (testing if this is OFF)...")
    GPIO.output(SQUIRT_PIN, GPIO.LOW)
    
    # Small delay to make sure it takes effect
    time.sleep(0.5)
    
    print("GPIO initialized, relay state should have changed")
    print("Script will exit. Check if relay is ON or OFF now.")
finally:
    # Don't run GPIO.cleanup() to see if the pin state persists
    print("Exiting without GPIO cleanup to maintain pin state.") 