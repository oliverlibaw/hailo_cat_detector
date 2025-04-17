#!/usr/bin/env python3
"""
Modified version of test_relays.py for debugging the squirt relay issue.
This script sets up GPIO differently and provides more direct control.
"""

import os
import sys
import time

try:
    import RPi.GPIO as GPIO
    print("Successfully imported GPIO module")
except ImportError as e:
    print(f"Error importing GPIO module: {e}")
    print("This script must be run on a Raspberry Pi with RPi.GPIO installed.")
    print("If you're testing on the Pi, make sure you're running with sudo privileges.")
    sys.exit(1)

# GPIO Pin Setup
SQUIRT_PIN = 5     # Squirt relay (triggers water gun)
LEFT_PIN = 13      # Left relay (triggers for left-side movement)
RIGHT_PIN = 6      # Right relay (triggers for right-side movement)

# Print at every step for more verbose debugging
def debug_print(message):
    print(f"DEBUG: {message}")

def setup_gpio():
    """Initialize GPIO with maximum debug information"""
    debug_print("Setting up GPIO...")
    
    # First, check if GPIO is already in use
    GPIO.setwarnings(False)  # Suppress warnings for now
    
    # Set the GPIO mode
    debug_print("Setting GPIO mode to BCM")
    GPIO.setmode(GPIO.BCM)
    
    # Setup pins one by one with thorough debug info
    debug_print(f"Setting up SQUIRT relay pin {SQUIRT_PIN}...")
    GPIO.setup(SQUIRT_PIN, GPIO.OUT)
    debug_print("SQUIRT pin setup complete")
    
    # Default squirt pin to HIGH - testing if this keeps it OFF
    debug_print(f"Setting SQUIRT pin {SQUIRT_PIN} to HIGH (assuming OFF)...")
    GPIO.output(SQUIRT_PIN, GPIO.HIGH)
    debug_print("SQUIRT pin set to HIGH")
    
    # Setup other pins
    debug_print(f"Setting up LEFT relay pin {LEFT_PIN}...")
    GPIO.setup(LEFT_PIN, GPIO.OUT)
    GPIO.output(LEFT_PIN, GPIO.HIGH)  # Assuming HIGH = OFF for active LOW relays
    debug_print("LEFT pin setup complete and set to HIGH")
    
    debug_print(f"Setting up RIGHT relay pin {RIGHT_PIN}...")
    GPIO.setup(RIGHT_PIN, GPIO.OUT)
    GPIO.output(RIGHT_PIN, GPIO.HIGH)  # Assuming HIGH = OFF for active LOW relays
    debug_print("RIGHT pin setup complete and set to HIGH")
    
    debug_print("All pins initialized")
    time.sleep(1)  # Longer delay to ensure setup takes effect

def cleanup_gpio():
    """Clean up GPIO resources with thorough debug info"""
    debug_print("Starting GPIO cleanup...")
    
    # First set all pins to HIGH (assuming this is OFF)
    debug_print(f"Setting SQUIRT pin {SQUIRT_PIN} to HIGH before cleanup...")
    GPIO.output(SQUIRT_PIN, GPIO.HIGH)
    
    debug_print(f"Setting LEFT pin {LEFT_PIN} to HIGH before cleanup...")
    GPIO.output(LEFT_PIN, GPIO.HIGH)
    
    debug_print(f"Setting RIGHT pin {RIGHT_PIN} to HIGH before cleanup...")
    GPIO.output(RIGHT_PIN, GPIO.HIGH)
    
    time.sleep(0.5)  # Give pins time to update
    
    debug_print("Running GPIO.cleanup()...")
    GPIO.cleanup()
    debug_print("GPIO cleanup complete")

def toggle_pin(pin, duration, state_high):
    """Toggle a pin to the specified state for a duration"""
    current_state = "HIGH" if state_high else "LOW"
    debug_print(f"Setting pin {pin} to {current_state}...")
    GPIO.output(pin, GPIO.HIGH if state_high else GPIO.LOW)
    
    debug_print(f"Waiting {duration} seconds...")
    time.sleep(duration)
    
    opposite_state = "HIGH" if not state_high else "LOW"
    debug_print(f"Setting pin {pin} back to {opposite_state}...")
    GPIO.output(pin, GPIO.HIGH if not state_high else GPIO.LOW)

def main():
    """Main interactive testing function"""
    try:
        setup_gpio()
        
        print("\n=== Debug Relay Test ===")
        print("Commands:")
        print("  s1 - Set squirt relay to HIGH (should be OFF)")
        print("  s0 - Set squirt relay to LOW (should be ON)")
        print("  l1 - Set left relay to HIGH")
        print("  l0 - Set left relay to LOW")
        print("  r1 - Set right relay to HIGH")
        print("  r0 - Set right relay to LOW")
        print("  q - Quit")
        
        while True:
            command = input("\nEnter command: ").lower().strip()
            
            if command == 'q':
                debug_print("User requested quit")
                break
            
            elif command == 's1':
                debug_print("Setting squirt relay to HIGH")
                GPIO.output(SQUIRT_PIN, GPIO.HIGH)
                
            elif command == 's0':
                debug_print("Setting squirt relay to LOW")
                GPIO.output(SQUIRT_PIN, GPIO.LOW)
                
            elif command == 'l1':
                debug_print("Setting left relay to HIGH")
                GPIO.output(LEFT_PIN, GPIO.HIGH)
                
            elif command == 'l0':
                debug_print("Setting left relay to LOW")
                GPIO.output(LEFT_PIN, GPIO.LOW)
                
            elif command == 'r1':
                debug_print("Setting right relay to HIGH")
                GPIO.output(RIGHT_PIN, GPIO.HIGH)
                
            elif command == 'r0':
                debug_print("Setting right relay to LOW")
                GPIO.output(RIGHT_PIN, GPIO.LOW)
                
            else:
                print("Unknown command. Please use s1, s0, l1, l0, r1, r0, or q.")
    
    except KeyboardInterrupt:
        debug_print("Program interrupted by user")
    except Exception as e:
        debug_print(f"Error: {e}")
    finally:
        cleanup_gpio()
        print("Program terminated")

if __name__ == "__main__":
    main() 