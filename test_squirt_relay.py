#!/usr/bin/env python3
"""
Simple test script specifically for the squirt relay.
Tests direct GPIO control with both HIGH and LOW states.
"""

import RPi.GPIO as GPIO
import time
import sys

# SQUIRT relay pin - based on the main script
SQUIRT_PIN = 5

def setup():
    """Initialize GPIO"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SQUIRT_PIN, GPIO.OUT)
    print(f"GPIO pin {SQUIRT_PIN} initialized")

def cleanup():
    """Clean up GPIO resources"""
    print("Cleaning up GPIO...")
    GPIO.cleanup()
    print("GPIO cleanup completed")

def main():
    try:
        setup()
        
        # Start with OFF state (initial assumption: HIGH = OFF)
        print(f"\nSetting pin {SQUIRT_PIN} to HIGH...")
        GPIO.output(SQUIRT_PIN, GPIO.HIGH)
        
        print("\nIs the relay OFF now? (y/n)")
        response = input("> ").lower().strip()
        
        if response == 'y':
            print("Confirmed: HIGH = OFF for this relay")
            off_state = GPIO.HIGH
            on_state = GPIO.LOW
        else:
            print("Noted: HIGH = ON for this relay")
            off_state = GPIO.LOW
            on_state = GPIO.HIGH
        
        # Now cycle through states based on user input
        while True:
            print("\nCommands:")
            print("  1 - Turn relay ON")
            print("  0 - Turn relay OFF")
            print("  q - Quit")
            
            cmd = input("> ").strip().lower()
            
            if cmd == '1':
                print(f"Setting pin {SQUIRT_PIN} to {'LOW' if on_state == GPIO.LOW else 'HIGH'}...")
                GPIO.output(SQUIRT_PIN, on_state)
                print("Relay should now be ON")
            
            elif cmd == '0':
                print(f"Setting pin {SQUIRT_PIN} to {'LOW' if off_state == GPIO.LOW else 'HIGH'}...")
                GPIO.output(SQUIRT_PIN, off_state)
                print("Relay should now be OFF")
            
            elif cmd == 'q':
                # Make sure to turn off before quitting
                GPIO.output(SQUIRT_PIN, off_state)
                print("Turning relay OFF and quitting...")
                break
            
            else:
                print("Unknown command. Please use 1, 0, or q.")
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        cleanup()

if __name__ == "__main__":
    main() 