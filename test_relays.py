#!/usr/bin/env python3
"""
Simple script to test the relay connections interactively.
Allows manual control of each relay via keyboard inputs:
- l: Activate "move left" relay
- r: Activate "move right" relay
- s: Activate "squirt" relay
- q: Quit the program
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

# GPIO Pin Setup - using the same pin definitions as the main script
RELAY_PINS = {
    'squirt': 5,    # Squirt relay (triggers water gun)
    'left': 6,      # Left relay (triggers for left-side movement)
    'right': 13,    # Right relay (triggers for right-side movement)
    'unused': 15    # Unused relay
}

# Important: Set to True if your relay module activates on LOW rather than HIGH
RELAY_ACTIVE_LOW = True    # Many relay HATs activate on LOW signal

# This flag indicates relays are "normally closed" - they're ON when not activated
RELAY_NORMALLY_CLOSED = True  # Set to True if relays are ON by default and turn OFF when activated

# Duration for each activation (seconds)
DEFAULT_DURATION = 0.5

def setup_gpio():
    """Initialize GPIO pins for relays"""
    GPIO.setmode(GPIO.BCM)
    for pin in RELAY_PINS.values():
        GPIO.setup(pin, GPIO.OUT)
        
    # Start with all relays turned OFF
    for pin in RELAY_PINS.values():
        set_relay(pin, False)
        
    print("GPIO pins initialized, all relays set to OFF state")

def cleanup_gpio():
    """Clean up GPIO resources"""
    print("Turning off all relays...")
    for pin in RELAY_PINS.values():
        set_relay(pin, False)
    GPIO.cleanup()
    print("GPIO cleanup completed")

def set_relay(pin, state):
    """
    Set relay state (True = ON, False = OFF)
    
    For normally closed relays:
    - When state=True (ON), we want to DEACTIVATE the relay (to open the circuit)
    - When state=False (OFF), we want to ACTIVATE the relay (to close the circuit)
    """
    # If relays are normally closed, invert the state
    actual_state = not state if RELAY_NORMALLY_CLOSED else state
    
    # Invert the signal if relays are active LOW
    gpio_state = GPIO.LOW if (actual_state and RELAY_ACTIVE_LOW) or (not actual_state and not RELAY_ACTIVE_LOW) else GPIO.HIGH
    
    # Print the exact GPIO state being set for debugging
    print(f"Setting GPIO pin {pin} to {'LOW' if gpio_state == GPIO.LOW else 'HIGH'}")
    
    # Set the GPIO pin state
    GPIO.output(pin, gpio_state)
    
    # Return the physical relay state for clarity
    return "OPEN" if state else "CLOSED" if RELAY_NORMALLY_CLOSED else "CLOSED" if state else "OPEN"

def activate_relay_with_duration(pin, duration, name):
    """Activate a relay for a specific duration and then turn it off"""
    try:
        # Turn on the relay
        print(f"Activating {name} relay (pin {pin}) for {duration} seconds...")
        relay_state = set_relay(pin, True)
        print(f"{name} relay now physically {relay_state}")
        
        # Keep it on for the duration
        time.sleep(duration)
        
        # Turn off the relay
        relay_state = set_relay(pin, False)
        print(f"{name} relay now physically {relay_state}")
        
    except KeyboardInterrupt:
        # Handle Ctrl+C
        set_relay(pin, False)
        print(f"\nInterrupted! {name} relay turned OFF")
        raise

def relay_test():
    """Interactive relay testing function"""
    try:
        setup_gpio()
        
        print("\n=== Relay Test Script (Enhanced) ===")
        print(f"Relay configuration: {'ACTIVE LOW' if RELAY_ACTIVE_LOW else 'ACTIVE HIGH'}, {'NORMALLY CLOSED' if RELAY_NORMALLY_CLOSED else 'NORMALLY OPEN'}")
        print("\nCommands:")
        print("  l - Activate LEFT movement relay (pin {})".format(RELAY_PINS['left']))
        print("  r - Activate RIGHT movement relay (pin {})".format(RELAY_PINS['right']))
        print("  s - Activate SQUIRT relay (pin {})".format(RELAY_PINS['squirt']))
        print("  a - Activate ALL relays")
        print("  0 - Turn OFF all relays")
        print("  d - Toggle relay activation duration (current: {}s)".format(DEFAULT_DURATION))
        print("  t - Test all relays in sequence")
        print("  i - Inverse test (activates each relay by turning others ON)")
        print("  q - Quit the program")
        
        duration = DEFAULT_DURATION
        
        while True:
            command = input(f"\nEnter command (l/r/s/a/0/d/t/i/q) [duration={duration}s]: ").lower().strip()
            
            if command == 'q':
                print("Exiting program...")
                break
                
            elif command == 'l':
                # Left relay test
                activate_relay_with_duration(RELAY_PINS['left'], duration, "LEFT")
                
            elif command == 'r':
                # Right relay test
                activate_relay_with_duration(RELAY_PINS['right'], duration, "RIGHT")
                
            elif command == 's':
                # Squirt relay test
                activate_relay_with_duration(RELAY_PINS['squirt'], duration, "SQUIRT")
                
            elif command == 'a':
                # Turn on all relays
                print("Activating ALL relays...")
                for name, pin in RELAY_PINS.items():
                    if name != 'unused':
                        state = set_relay(pin, True)
                        print(f"{name.upper()} relay (pin {pin}) physically {state}")
                
                print(f"Keeping ALL relays ON for {duration} seconds...")
                time.sleep(duration)
                
                # Turn off all relays
                print("Deactivating ALL relays...")
                for name, pin in RELAY_PINS.items():
                    if name != 'unused':
                        state = set_relay(pin, False)
                        print(f"{name.upper()} relay (pin {pin}) physically {state}")
                
            elif command == '0':
                # Turn off all relays
                for name, pin in RELAY_PINS.items():
                    if name != 'unused':
                        state = set_relay(pin, False)
                        print(f"{name.upper()} relay (pin {pin}) physically {state}")
                print("All relays turned OFF")
            
            elif command == 'd':
                # Change duration
                try:
                    new_duration = float(input(f"Enter new duration in seconds (current: {duration}s): "))
                    if new_duration > 0:
                        duration = new_duration
                        print(f"Duration set to {duration} seconds")
                    else:
                        print("Duration must be positive. Using current value.")
                except ValueError:
                    print("Invalid input. Using current duration.")
            
            elif command == 't':
                # Test all relays in sequence
                print("\n=== Testing all relays in sequence ===")
                
                # First make sure all relays are OFF
                for pin in RELAY_PINS.values():
                    set_relay(pin, False)
                print("All relays turned OFF")
                time.sleep(1)
                
                # Test each relay one by one
                for name, pin in RELAY_PINS.items():
                    if name != 'unused':
                        print(f"\nTesting {name.upper()} relay (pin {pin})...")
                        activate_relay_with_duration(pin, duration, name.upper())
                        time.sleep(0.5)
                
                print("\nSequential relay test completed")
            
            elif command == 'i':
                # Inverse test (turn on all except one)
                print("\n=== Inverse relay test ===")
                print("This will turn ON all relays EXCEPT the specified one")
                print("(This might help if your relays are connected in reverse)")
                
                # Turn on all relays first
                for pin in RELAY_PINS.values():
                    if pin != RELAY_PINS['unused']:
                        set_relay(pin, True)
                print("All relays turned ON")
                time.sleep(1)
                
                # Now test by turning each relay OFF one at a time
                for name, pin in RELAY_PINS.items():
                    if name != 'unused':
                        print(f"\nDeactivating only {name.upper()} relay (pin {pin})...")
                        set_relay(pin, False)
                        print(f"Waiting {duration} seconds...")
                        time.sleep(duration)
                        set_relay(pin, True)
                        time.sleep(0.5)
                
                # Turn everything off at the end
                for pin in RELAY_PINS.values():
                    set_relay(pin, False)
                print("\nInverse relay test completed, all relays OFF")
                
            else:
                print("Unknown command. Please use l, r, s, a, 0, d, t, i, or q.")
                
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Cleaning up...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        cleanup_gpio()
        print("Program terminated")

if __name__ == "__main__":
    relay_test() 