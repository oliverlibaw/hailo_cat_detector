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
    'squirt': 6,    # Center relay (triggers squirt)
    'left': 5,      # Left relay (triggers for left-side movement)
    'right': 13,    # Right relay (triggers for right-side movement)
    'unused': 15    # Unused relay
}

# Important: Set to True if your relay module activates on LOW rather than HIGH
RELAY_ACTIVE_LOW = True    # Many relay HATs activate on LOW signal

# This flag indicates relays are "normally closed" - they're ON when not activated
RELAY_NORMALLY_CLOSED = True  # Set to True if relays are ON by default and turn OFF when activated

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
    
    # Set the GPIO pin state
    GPIO.output(pin, gpio_state)
    
    # Return the physical relay state for clarity
    return "OPEN" if state else "CLOSED" if RELAY_NORMALLY_CLOSED else "CLOSED" if state else "OPEN"

def relay_test():
    """Interactive relay testing function"""
    try:
        setup_gpio()
        
        print("\n=== Relay Test Script ===")
        print(f"Relay configuration: {'ACTIVE LOW' if RELAY_ACTIVE_LOW else 'ACTIVE HIGH'}, {'NORMALLY CLOSED' if RELAY_NORMALLY_CLOSED else 'NORMALLY OPEN'}")
        print("\nCommands:")
        print("  l - Activate LEFT movement relay (pin {})".format(RELAY_PINS['left']))
        print("  r - Activate RIGHT movement relay (pin {})".format(RELAY_PINS['right']))
        print("  s - Activate SQUIRT relay (pin {})".format(RELAY_PINS['squirt']))
        print("  a - Activate ALL relays")
        print("  0 - Turn OFF all relays")
        print("  q - Quit the program")
        print("\nActive relays will remain ON until turned off or program exit.")
        
        while True:
            command = input("\nEnter command (l/r/s/a/0/q): ").lower().strip()
            
            if command == 'q':
                print("Exiting program...")
                break
                
            elif command == 'l':
                # Turn on left relay, turn off others
                relay_state = set_relay(RELAY_PINS['left'], True)
                set_relay(RELAY_PINS['right'], False)
                set_relay(RELAY_PINS['squirt'], False)
                print(f"LEFT relay activated (pin {RELAY_PINS['left']}) - Relay physically {relay_state}")
                
            elif command == 'r':
                # Turn on right relay, turn off others
                relay_state = set_relay(RELAY_PINS['right'], True)
                set_relay(RELAY_PINS['left'], False)
                set_relay(RELAY_PINS['squirt'], False)
                print(f"RIGHT relay activated (pin {RELAY_PINS['right']}) - Relay physically {relay_state}")
                
            elif command == 's':
                # Turn on squirt relay, turn off others
                relay_state = set_relay(RELAY_PINS['squirt'], True)
                set_relay(RELAY_PINS['left'], False)
                set_relay(RELAY_PINS['right'], False)
                print(f"SQUIRT relay activated (pin {RELAY_PINS['squirt']}) - Relay physically {relay_state}")
                
            elif command == 'a':
                # Turn on all relays
                states = []
                for name, pin in RELAY_PINS.items():
                    if name != 'unused':
                        state = set_relay(pin, True)
                        states.append(f"{name}={state}")
                print(f"ALL relays activated: {', '.join(states)}")
                
            elif command == '0':
                # Turn off all relays
                for pin in RELAY_PINS.values():
                    set_relay(pin, False)
                print("All relays turned OFF")
                
            else:
                print("Unknown command. Please use l, r, s, a, 0, or q.")
                
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Cleaning up...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        cleanup_gpio()
        print("Program terminated")

if __name__ == "__main__":
    relay_test() 