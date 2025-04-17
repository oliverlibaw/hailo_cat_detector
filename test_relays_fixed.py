#!/usr/bin/env python3
"""
Simplified relay test script with the correct pin states based on testing.
This script corrects the squirt relay behavior.
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

# GPIO Pin Setup - simplified
RELAY_PINS = {
    'squirt': 5,    # Squirt relay - pin 5 (BCM)
    'left': 13,     # Left relay - pin 13 (BCM)
    'right': 6      # Right relay - pin 6 (BCM)
}

# IMPORTANT: The squirt relay has opposite behavior from the other relays
# For squirt relay: HIGH = ON, LOW = OFF
# For other relays: LOW = ON, HIGH = OFF (standard active-low relay behavior)

def setup_gpio():
    """Initialize GPIO pins for relays with the correct states"""
    print("Setting up GPIO...")
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    # Initialize all pins and set to OFF state
    for name, pin in RELAY_PINS.items():
        GPIO.setup(pin, GPIO.OUT)
        
        # Set the correct initial state for each relay type
        if name == 'squirt':
            # Squirt relay is OFF when LOW
            print(f"Setting {name} relay (pin {pin}) to LOW (OFF state)...")
            GPIO.output(pin, GPIO.LOW)
        else:
            # Other relays are OFF when HIGH (active-low)
            print(f"Setting {name} relay (pin {pin}) to HIGH (OFF state)...")
            GPIO.output(pin, GPIO.HIGH)
    
    print("All relays initialized to OFF state")
    time.sleep(0.5)  # Small delay to ensure states take effect

def cleanup_gpio():
    """Clean up GPIO resources"""
    print("Turning off all relays...")
    
    # Set all relays to OFF state before cleanup
    for name, pin in RELAY_PINS.items():
        if name == 'squirt':
            # Squirt relay is OFF when LOW
            GPIO.output(pin, GPIO.LOW)
        else:
            # Other relays are OFF when HIGH
            GPIO.output(pin, GPIO.HIGH)
    
    time.sleep(0.5)  # Give time for states to take effect
    GPIO.cleanup()
    print("GPIO cleanup completed")

def set_relay(name, state):
    """
    Set relay state (True = ON, False = OFF) with the correct logic
    for each relay type
    """
    pin = RELAY_PINS[name]
    
    if name == 'squirt':
        # Squirt relay: HIGH = ON, LOW = OFF
        gpio_state = GPIO.HIGH if state else GPIO.LOW
    else:
        # Standard relays: LOW = ON, HIGH = OFF (active-low)
        gpio_state = GPIO.LOW if state else GPIO.HIGH
    
    print(f"Setting {name} relay (pin {pin}) to {'HIGH' if gpio_state == GPIO.HIGH else 'LOW'}")
    GPIO.output(pin, gpio_state)
    return f"{'ON' if state else 'OFF'}"

def activate_relay(name, duration):
    """Activate a relay for a specific duration and then turn it off"""
    print(f"Activating {name} relay for {duration} seconds...")
    
    # Turn ON
    status = set_relay(name, True)
    print(f"{name} relay is now {status}")
    
    # Wait for duration
    time.sleep(duration)
    
    # Turn OFF
    status = set_relay(name, False)
    print(f"{name} relay is now {status}")

def main():
    """Interactive relay testing function"""
    try:
        setup_gpio()
        
        print("\n=== Relay Test (Fixed) ===")
        print("Commands:")
        print("  l - Activate LEFT relay")
        print("  r - Activate RIGHT relay")
        print("  s - Activate SQUIRT relay")
        print("  a - Activate ALL relays")
        print("  0 - Turn OFF all relays")
        print("  q - Quit")
        
        duration = 1.0  # Default duration in seconds
        
        while True:
            command = input(f"\nEnter command (l/r/s/a/0/q) [duration={duration}s]: ").lower().strip()
            
            if command == 'q':
                print("Exiting program...")
                break
                
            elif command == 'l':
                activate_relay('left', duration)
                
            elif command == 'r':
                activate_relay('right', duration)
                
            elif command == 's':
                activate_relay('squirt', duration)
                
            elif command == 'a':
                # Turn on all relays
                print(f"Activating ALL relays for {duration} seconds...")
                for name in RELAY_PINS.keys():
                    set_relay(name, True)
                
                time.sleep(duration)
                
                # Turn off all relays
                print("Turning OFF all relays...")
                for name in RELAY_PINS.keys():
                    set_relay(name, False)
                
            elif command == '0':
                # Turn off all relays
                print("Turning OFF all relays...")
                for name in RELAY_PINS.keys():
                    set_relay(name, False)
                
            elif command.replace('.', '', 1).isdigit():
                # Change duration if a number was entered
                try:
                    new_duration = float(command)
                    if new_duration > 0:
                        duration = new_duration
                        print(f"Duration set to {duration} seconds")
                    else:
                        print("Duration must be positive")
                except ValueError:
                    print("Invalid duration")
            else:
                print("Unknown command. Please use l, r, s, a, 0, or q.")
                
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        cleanup_gpio()
        print("Program terminated")

if __name__ == "__main__":
    main() 