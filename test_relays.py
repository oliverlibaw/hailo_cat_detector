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
    'squirt': 5,     # Squirt relay (triggers water gun) - Channel 1
    'left': 13,      # Left relay (triggers for left-side movement)
    'right': 6,    # Right relay (triggers for right-side movement)
    'unused': 16    # Unused relay - Channel 4
}

# Important: Set to True if your relay module activates on LOW rather than HIGH
RELAY_ACTIVE_LOW = True    # Many relay HATs activate on LOW signal

# This flag indicates relays are "normally closed" - they're ON when not activated
RELAY_NORMALLY_CLOSED = False  # Changed to False since we're hearing the relay click

# The squirt relay needs separate handling
SQUIRT_RELAY_ON_STATE = GPIO.HIGH   # Set to HIGH to turn the relay ON
SQUIRT_RELAY_OFF_STATE = GPIO.LOW   # Set to LOW to turn the relay OFF

# Enable debug mode for detailed relay states
DEBUG_MODE = True

# Duration for each activation (seconds)
DEFAULT_DURATION = 1.0  # Increased default duration to 1.0 seconds

def setup_gpio():
    """Initialize GPIO pins for relays"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)  # Suppress warnings
    
    print("Initializing GPIO pins...")
    
    # First, set default pin states to avoid any immediate activation
    for pin in RELAY_PINS.values():
        GPIO.setup(pin, GPIO.OUT)
    
    # Initialize the squirt relay to OFF state first (explicitly)
    print(f"Setting SQUIRT relay (pin {RELAY_PINS['squirt']}) to OFF state...")
    GPIO.output(RELAY_PINS['squirt'], SQUIRT_RELAY_OFF_STATE)
    print(f"SQUIRT relay initialized to {'LOW' if SQUIRT_RELAY_OFF_STATE == GPIO.LOW else 'HIGH'}")
    
    # Initialize other relays
    print("Initializing other relays to OFF state...")
    for name, pin in RELAY_PINS.items():
        if name != 'unused' and name != 'squirt':
            # For Active LOW relays, initialize pins to proper OFF state
            if RELAY_ACTIVE_LOW:
                init_state = GPIO.HIGH  # Initialize to inactive state (HIGH = OFF for active low)
            else:
                init_state = GPIO.LOW   # Initialize to inactive state (LOW = OFF for active high)
                
            GPIO.output(pin, init_state)
            print(f"{name.upper()} relay (pin {pin}) initialized to {'LOW' if init_state == GPIO.LOW else 'HIGH'}")
    
    # Small delay to ensure initial states take effect
    time.sleep(0.5)
    
    # Now properly set all relays to OFF using our logic
    print("Final verification of all relays to OFF state...")
    for name, pin in RELAY_PINS.items():
        if name != 'unused':
            print(f"Setting {name.upper()} relay (pin {pin}) to OFF state...")
            set_relay(pin, False)
    
    print("GPIO pins initialized, all relays should be OFF")
    # Small delay to ensure changes take effect
    time.sleep(0.5)

def cleanup_gpio():
    """Clean up GPIO resources"""
    print("Turning off all relays...")
    
    # First turn off the squirt relay directly
    print(f"Turning off SQUIRT relay (pin {RELAY_PINS['squirt']})...")
    GPIO.output(RELAY_PINS['squirt'], SQUIRT_RELAY_OFF_STATE)
    
    # Then turn off other relays
    for name, pin in RELAY_PINS.items():
        if name != 'unused' and name != 'squirt':
            set_relay(pin, False)
            print(f"{name.upper()} relay (pin {pin}) turned OFF")
    
    # Small delay to ensure changes take effect
    time.sleep(0.5)
    
    GPIO.cleanup()
    print("GPIO cleanup completed")

def set_relay(pin, state):
    """
    Set relay state (True = ON, False = OFF)
    
    For normally closed relays:
    - When state=True (ON), we want to DEACTIVATE the relay (to open the circuit)
    - When state=False (OFF), we want to ACTIVATE the relay (to close the circuit)
    """
    # Special handling for squirt relay
    if pin == RELAY_PINS['squirt']:
        # Directly set the pin state for the squirt relay
        gpio_state = SQUIRT_RELAY_ON_STATE if state else SQUIRT_RELAY_OFF_STATE
        if DEBUG_MODE:
            print(f"SPECIAL: Using direct state for SQUIRT relay: {'LOW' if gpio_state == GPIO.LOW else 'HIGH'}")
    else:
        # Standard relay logic for other relays
        # If relays are normally closed, invert the state
        actual_state = not state if RELAY_NORMALLY_CLOSED else state
        
        # Invert the signal if relays are active LOW
        gpio_state = GPIO.LOW if (actual_state and RELAY_ACTIVE_LOW) or (not actual_state and not RELAY_ACTIVE_LOW) else GPIO.HIGH
    
    # Print the exact GPIO state being set for debugging
    if DEBUG_MODE:
        relay_name = "UNKNOWN"
        for name, p in RELAY_PINS.items():
            if p == pin:
                relay_name = name.upper()
        
        print(f"Setting {relay_name} relay (GPIO pin {pin}) to {'LOW' if gpio_state == GPIO.LOW else 'HIGH'}")
        print(f"  - State requested: {state} (ON if True, OFF if False)")
        if pin != RELAY_PINS['squirt']:
            print(f"  - Normally closed: {RELAY_NORMALLY_CLOSED}")
            print(f"  - Active low: {RELAY_ACTIVE_LOW}")
            print(f"  - Actual state after adjustments: {actual_state}")
    else:
        print(f"Setting GPIO pin {pin} to {'LOW' if gpio_state == GPIO.LOW else 'HIGH'}")
    
    # Set the GPIO pin state
    GPIO.output(pin, gpio_state)
    
    # Return the physical relay state for clarity
    if pin == RELAY_PINS['squirt']:
        return "CLOSED" if state else "OPEN"  # For squirt relay
    else:
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

def direct_gpio_control(pin, state):
    """
    Set GPIO pin directly to HIGH or LOW without any logic inversion.
    This bypasses all the relay logic for direct testing.
    
    Args:
        pin: GPIO pin number
        state: True for HIGH, False for LOW
    """
    gpio_state = GPIO.HIGH if state else GPIO.LOW
    print(f"DIRECT GPIO CONTROL: Setting pin {pin} directly to {'HIGH' if state else 'LOW'}")
    GPIO.output(pin, gpio_state)
    return gpio_state

def relay_test():
    """Interactive relay testing function"""
    try:
        global DEBUG_MODE, RELAY_ACTIVE_LOW  # Make these variables global
        setup_gpio()
        
        # Extra check for squirt relay
        print("\nExtra verification for SQUIRT relay...")
        print(f"Setting SQUIRT relay (pin {RELAY_PINS['squirt']}) to OFF directly...")
        GPIO.output(RELAY_PINS['squirt'], SQUIRT_RELAY_OFF_STATE)
        time.sleep(0.5)
        
        # Double-check that all relays are off at startup
        print("\nVerifying all relays are OFF...")
        for name, pin in RELAY_PINS.items():
            if name != 'unused':
                # For squirt relay, use direct control
                if name == 'squirt':
                    GPIO.output(pin, SQUIRT_RELAY_OFF_STATE)
                else:
                    # Force all other relays to OFF state again
                    set_relay(pin, False)
        
        print("\n=== Relay Test Script (Enhanced) ===")
        print(f"Relay configuration: {'ACTIVE LOW' if RELAY_ACTIVE_LOW else 'ACTIVE HIGH'}, {'NORMALLY CLOSED' if RELAY_NORMALLY_CLOSED else 'NORMALLY OPEN'}")
        print(f"Special squirt relay handling: ON={SQUIRT_RELAY_ON_STATE}, OFF={SQUIRT_RELAY_OFF_STATE}")
        print("\nCommands:")
        print("  l - Activate LEFT movement relay (pin {})".format(RELAY_PINS['left']))
        print("  r - Activate RIGHT movement relay (pin {})".format(RELAY_PINS['right']))
        print("  s - Activate SQUIRT relay (pin {})".format(RELAY_PINS['squirt']))
        print("  a - Activate ALL relays")
        print("  0 - Turn OFF all relays")
        print("  d - Toggle relay activation duration (current: {}s)".format(DEFAULT_DURATION))
        print("  t - Test all relays in sequence")
        print("  i - Inverse test (activates each relay by turning others ON)")
        print("  m - Toggle debug mode (current: {})".format(DEBUG_MODE))
        print("  p - Pulse test (rapidly toggle relay on/off)")
        print("  g - Direct GPIO control for testing")
        print("  q - Quit the program")
        
        duration = DEFAULT_DURATION
        
        while True:
            command = input(f"\nEnter command (l/r/s/a/0/d/t/i/m/p/g/q) [duration={duration}s]: ").lower().strip()
            
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
                        if name == 'squirt':
                            # Direct control for squirt relay
                            GPIO.output(pin, SQUIRT_RELAY_ON_STATE)
                            print(f"SQUIRT relay (pin {pin}) set to {'LOW' if SQUIRT_RELAY_ON_STATE == GPIO.LOW else 'HIGH'} (ON)")
                        else:
                            # Standard control for other relays
                            state = set_relay(pin, True)
                            print(f"{name.upper()} relay (pin {pin}) physically {state}")
                
                print(f"Keeping ALL relays ON for {duration} seconds...")
                time.sleep(duration)
                
                # Turn off all relays
                print("Deactivating ALL relays...")
                for name, pin in RELAY_PINS.items():
                    if name != 'unused':
                        if name == 'squirt':
                            # Direct control for squirt relay
                            GPIO.output(pin, SQUIRT_RELAY_OFF_STATE)
                            print(f"SQUIRT relay (pin {pin}) set to {'LOW' if SQUIRT_RELAY_OFF_STATE == GPIO.LOW else 'HIGH'} (OFF)")
                        else:
                            # Standard control for other relays
                            state = set_relay(pin, False)
                            print(f"{name.upper()} relay (pin {pin}) physically {state}")
            
            elif command == '0':
                # Turn off all relays
                for name, pin in RELAY_PINS.items():
                    if name != 'unused':
                        if name == 'squirt':
                            # Direct control for squirt relay
                            GPIO.output(pin, SQUIRT_RELAY_OFF_STATE)
                            print(f"SQUIRT relay (pin {pin}) set to {'LOW' if SQUIRT_RELAY_OFF_STATE == GPIO.LOW else 'HIGH'} (OFF)")
                        else:
                            # Standard control for other relays
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
                
            elif command == 'm':
                # Toggle debug mode
                DEBUG_MODE = not DEBUG_MODE
                print(f"Debug mode {'enabled' if DEBUG_MODE else 'disabled'}")
                
            elif command == 'p':
                # Pulse test for squirt relay
                relay_name = input("Enter relay to pulse test (l/r/s): ").lower().strip()
                if relay_name == 'l':
                    pin = RELAY_PINS['left']
                    name = "LEFT"
                elif relay_name == 'r':
                    pin = RELAY_PINS['right']
                    name = "RIGHT"
                elif relay_name == 's':
                    pin = RELAY_PINS['squirt']
                    name = "SQUIRT"
                else:
                    print("Invalid relay selection.")
                    continue
                
                pulse_count = int(input(f"Enter number of pulses for {name} relay: ") or "5")
                pulse_duration = float(input(f"Enter pulse duration in seconds (e.g. 0.1): ") or "0.1")
                
                print(f"\nPulsing {name} relay {pulse_count} times with {pulse_duration}s pulses...")
                for i in range(pulse_count):
                    print(f"Pulse {i+1}/{pulse_count}")
                    set_relay(pin, True)
                    time.sleep(pulse_duration)
                    set_relay(pin, False)
                    time.sleep(pulse_duration)
                print(f"Pulse test completed for {name} relay")
                
            elif command == 'g':
                # Direct GPIO control for testing
                print("\n=== Direct GPIO Control ===")
                print("This will bypass all relay logic and directly set GPIO state")
                
                relay_name = input("Enter relay to control (l/r/s): ").lower().strip()
                if relay_name == 'l':
                    pin = RELAY_PINS['left']
                    name = "LEFT"
                elif relay_name == 'r':
                    pin = RELAY_PINS['right']
                    name = "RIGHT"
                elif relay_name == 's':
                    pin = RELAY_PINS['squirt']
                    name = "SQUIRT"
                else:
                    print("Invalid relay selection.")
                    continue
                
                state_input = input("Enter state (1 for HIGH, 0 for LOW): ").strip()
                if state_input == '1':
                    direct_gpio_control(pin, True)
                    print(f"Set {name} relay (pin {pin}) directly to HIGH")
                elif state_input == '0':
                    direct_gpio_control(pin, False)
                    print(f"Set {name} relay (pin {pin}) directly to LOW")
                else:
                    print("Invalid state. Please use 1 for HIGH or 0 for LOW.")
                
            else:
                print("Unknown command. Please use l, r, s, a, 0, d, t, i, m, p, g, or q.")
                
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Cleaning up...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        cleanup_gpio()
        print("Program terminated")

if __name__ == "__main__":
    relay_test() 