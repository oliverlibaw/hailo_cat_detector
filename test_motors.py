#!/usr/bin/env python3
"""
Script to test motor movement with longer relay activation times.
Allows testing motor movement patterns by activating relays for specified durations.
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

# Duration to keep the relay active (in seconds)
DEFAULT_DURATION = 3.0

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

def activate_relay_for_duration(pin, duration, relay_name):
    """Activate a relay for the specified duration, then turn it off"""
    try:
        # Turn on the relay
        state = set_relay(pin, True)
        print(f"{relay_name} relay ACTIVATED (pin {pin}) - Relay physically {state}")
        print(f"Keeping relay ON for {duration} seconds...")
        
        # Wait for the specified duration
        time.sleep(duration)
        
        # Turn off the relay
        state = set_relay(pin, False)
        print(f"{relay_name} relay DEACTIVATED - Relay physically {state}")
        
    except KeyboardInterrupt:
        # Handle Ctrl+C to stop during activation
        state = set_relay(pin, False)
        print(f"\nInterrupted! {relay_name} relay DEACTIVATED - Relay physically {state}")
        raise

def motor_test():
    """Test motors with longer activation durations"""
    try:
        setup_gpio()
        
        print("\n=== Motor Movement Test ===")
        print(f"Relay configuration: {'ACTIVE LOW' if RELAY_ACTIVE_LOW else 'ACTIVE HIGH'}, {'NORMALLY CLOSED' if RELAY_NORMALLY_CLOSED else 'NORMALLY OPEN'}")
        print("This script will activate the selected relay for a specified duration.")
        print("Use this to observe the full movement pattern of your motors.")
        
        while True:
            print("\nOptions:")
            print("  1 - Test LEFT motor movement")
            print("  2 - Test RIGHT motor movement")
            print("  3 - Test SQUIRT mechanism")
            print("  4 - Custom motor test")
            print("  5 - Quit program")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '5':
                print("Exiting program...")
                break
                
            # Get duration for options 1-3
            if choice in ['1', '2', '3']:
                try:
                    duration = float(input(f"Enter activation duration in seconds (default: {DEFAULT_DURATION}): ") or DEFAULT_DURATION)
                except ValueError:
                    print("Invalid duration. Using default.")
                    duration = DEFAULT_DURATION
            
            if choice == '1':
                # Test left motor
                activate_relay_for_duration(RELAY_PINS['left'], duration, "LEFT")
                
            elif choice == '2':
                # Test right motor
                activate_relay_for_duration(RELAY_PINS['right'], duration, "RIGHT")
                
            elif choice == '3':
                # Test squirt mechanism
                activate_relay_for_duration(RELAY_PINS['squirt'], duration, "SQUIRT")
                
            elif choice == '4':
                # Custom test sequence
                print("\n=== Custom Motor Test ===")
                print("This will activate each motor in sequence with customizable durations.")
                
                try:
                    left_duration = float(input("LEFT motor duration (seconds, 0 to skip): ") or 0)
                    right_duration = float(input("RIGHT motor duration (seconds, 0 to skip): ") or 0)
                    squirt_duration = float(input("SQUIRT mechanism duration (seconds, 0 to skip): ") or 0)
                    pause_between = float(input("Pause between activations (seconds): ") or 1)
                except ValueError:
                    print("Invalid input. Using defaults.")
                    left_duration, right_duration, squirt_duration, pause_between = 2, 2, 2, 1
                
                print("\nStarting custom test sequence...")
                
                if left_duration > 0:
                    activate_relay_for_duration(RELAY_PINS['left'], left_duration, "LEFT")
                    time.sleep(pause_between)
                    
                if right_duration > 0:
                    activate_relay_for_duration(RELAY_PINS['right'], right_duration, "RIGHT")
                    time.sleep(pause_between)
                    
                if squirt_duration > 0:
                    activate_relay_for_duration(RELAY_PINS['squirt'], squirt_duration, "SQUIRT")
                
                print("Custom test sequence completed")
                
            else:
                print("Invalid choice. Please enter a number from 1 to 5.")
                
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Cleaning up...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        cleanup_gpio()
        print("Program terminated")

if __name__ == "__main__":
    motor_test() 