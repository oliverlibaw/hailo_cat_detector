#!/usr/bin/env python3
"""
Relay Calibration Test Script for Hailo Cat Detector
This script allows direct testing of the relays to determine the correct mapping
between GPIO pins and physical movement directions.
"""

import RPi.GPIO as GPIO
import time
import sys

# GPIO Pin Setup - We'll test all possible configurations
RELAY_PINS = {
    'PIN_5': 5,
    'PIN_6': 6,
    'PIN_13': 13,
    'PIN_15': 15
}

# Define the valid pin numbers to test
VALID_PINS = [5, 6, 13, 15]

# Define possible relay states
RELAY_ACTIVE_LOW = True  # Set to match your relay module's behavior

def setup_gpio():
    """Initialize GPIO pins for relays."""
    GPIO.setmode(GPIO.BCM)  # Use Broadcom pin numbers
    GPIO.setwarnings(False)

    # Initialize all relay pins as outputs
    for pin in RELAY_PINS.values():
        GPIO.setup(pin, GPIO.OUT)
        # Initialize to inactive state
        off_state = GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW
        GPIO.output(pin, off_state)
        
    print("GPIO Initialized")
    return True

def activate_relay(pin, duration, active_low=True):
    """
    Activate a specific relay for a given duration (in seconds).
    """
    if pin not in VALID_PINS:
        print(f"ERROR: Pin {pin} is not in the list of valid pins {VALID_PINS}")
        print("Please choose one of the pins that was initialized")
        return False
        
    try:
        pin_name = "Unknown"
        for name, p in RELAY_PINS.items():
            if p == pin:
                pin_name = name
                break
                
        # Determine ON/OFF GPIO levels based on active_low setting
        on_state = GPIO.LOW if active_low else GPIO.HIGH
        off_state = GPIO.HIGH if active_low else GPIO.LOW
        
        print(f"Activating {pin_name} (PIN {pin}) for {duration:.2f}s with on_state={'LOW' if on_state == GPIO.LOW else 'HIGH'}")
        
        # Activate relay
        GPIO.output(pin, on_state)
        
        # Wait for specified duration
        time.sleep(duration)
        
        # Deactivate relay
        GPIO.output(pin, off_state)
        print(f"Deactivated {pin_name} (PIN {pin})")
        return True
        
    except Exception as e:
        print(f"Error activating relay on pin {pin}: {e}")
        # Ensure relay is turned off in case of error
        try:
            GPIO.output(pin, off_state)
        except Exception as inner_error:
            print(f"Additionally, could not reset pin state: {inner_error}")
        return False

def cleanup():
    """Clean up GPIO resources."""
    try:
        # Turn off all relays
        for pin in RELAY_PINS.values():
            try:
                off_state = GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW
                GPIO.output(pin, off_state)
            except:
                # Individual pin might not be set up, continue to others
                pass
            
        # Clean up GPIO resources
        GPIO.cleanup()
        print("GPIO cleaned up")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")

def interactive_test():
    """Interactive test mode for relays."""
    print("\n=== Relay Testing and Calibration ===")
    print("This tool will help determine which GPIO pin controls which direction.")
    print("Instructions:")
    print("1. Observe the direction the device moves when each relay is activated")
    print("2. Take notes on which pin moves the device in which direction")
    print("3. Update your main script with the correct pin assignments")
    
    gpio_setup = setup_gpio()
    if not gpio_setup:
        print("ERROR: Failed to set up GPIO. Cannot continue.")
        return
    
    try:
        while True:
            print("\nTest Options:")
            print("1 - Test PIN 5")
            print("2 - Test PIN 6")
            print("3 - Test PIN 13")
            print("4 - Test PIN 15")
            print("5 - Test Active High (ON=HIGH, OFF=LOW)")
            print("6 - Test Active Low (ON=LOW, OFF=HIGH)")
            print("7 - Rapid Left-Right Sequence Test")
            print("q - Quit")
            
            choice = input("\nEnter option: ").strip().lower()
            
            if choice == 'q':
                break
                
            if choice == '1':
                activate_relay(5, 1.0, RELAY_ACTIVE_LOW)
            elif choice == '2':
                activate_relay(6, 1.0, RELAY_ACTIVE_LOW)
            elif choice == '3':
                activate_relay(13, 1.0, RELAY_ACTIVE_LOW)
            elif choice == '4':
                activate_relay(15, 1.0, RELAY_ACTIVE_LOW)
            elif choice == '5':
                # Test active high
                try:
                    duration = 1.0
                    pin_input = input("Which pin to test as active HIGH? (5, 6, 13, or 15): ")
                    if not pin_input.isdigit() or int(pin_input) not in VALID_PINS:
                        print(f"Invalid pin number. Please use one of: {VALID_PINS}")
                        continue
                        
                    pin = int(pin_input)
                    print(f"Testing PIN {pin} as active HIGH (ON=HIGH, OFF=LOW)")
                    GPIO.output(pin, GPIO.HIGH)
                    time.sleep(duration)
                    GPIO.output(pin, GPIO.LOW)
                except Exception as e:
                    print(f"Error testing active HIGH: {e}")
            elif choice == '6':
                # Test active low
                try:
                    duration = 1.0
                    pin_input = input("Which pin to test as active LOW? (5, 6, 13, or 15): ")
                    if not pin_input.isdigit() or int(pin_input) not in VALID_PINS:
                        print(f"Invalid pin number. Please use one of: {VALID_PINS}")
                        continue
                        
                    pin = int(pin_input)
                    print(f"Testing PIN {pin} as active LOW (ON=LOW, OFF=HIGH)")
                    GPIO.output(pin, GPIO.LOW)
                    time.sleep(duration)
                    GPIO.output(pin, GPIO.HIGH)
                except Exception as e:
                    print(f"Error testing active LOW: {e}")
            elif choice == '7':
                # Test rapid left-right sequence
                try:
                    left_input = input("Enter pin for LEFT movement (5, 6, 13, or 15): ")
                    if not left_input.isdigit() or int(left_input) not in VALID_PINS:
                        print(f"Invalid pin number for LEFT. Please use one of: {VALID_PINS}")
                        continue
                        
                    right_input = input("Enter pin for RIGHT movement (5, 6, 13, or 15): ")
                    if not right_input.isdigit() or int(right_input) not in VALID_PINS:
                        print(f"Invalid pin number for RIGHT. Please use one of: {VALID_PINS}")
                        continue
                    
                    iterations_input = input("Number of iterations (default: 3): ")
                    iterations = 3  # Default value
                    if iterations_input.isdigit():
                        iterations = int(iterations_input)
                    
                    left_pin = int(left_input)
                    right_pin = int(right_input)
                    
                    print(f"Testing sequence: LEFT (PIN {left_pin}) then RIGHT (PIN {right_pin})")
                    for i in range(iterations):
                        print(f"Iteration {i+1}/{iterations}")
                        print("Moving LEFT...")
                        activate_relay(left_pin, 0.5, RELAY_ACTIVE_LOW)
                        time.sleep(0.5)
                        print("Moving RIGHT...")
                        activate_relay(right_pin, 0.5, RELAY_ACTIVE_LOW)
                        time.sleep(0.5)
                except Exception as e:
                    print(f"Error in sequence test: {e}")
            else:
                print("Invalid option")
                
    except KeyboardInterrupt:
        print("\nTest interrupted")
    finally:
        cleanup()

def automatic_test():
    """Run a predetermined sequence of relay activations."""
    gpio_setup = setup_gpio()
    if not gpio_setup:
        print("ERROR: Failed to set up GPIO. Cannot continue.")
        return
    
    try:
        print("\n=== Starting Automatic Relay Test Sequence ===")
        
        # Test each pin in sequence
        print("\nTesting individual pins:")
        for name, pin in RELAY_PINS.items():
            print(f"\nActivating {name} (PIN {pin})...")
            activate_relay(pin, 1.0, RELAY_ACTIVE_LOW)
            time.sleep(1.0)  # Pause between tests
            
        # Wait for user confirmation to continue
        input("\nPress Enter to continue to direction tests...")
        
        # Test combinations for directional control
        print("\nTesting PIN 6 as LEFT:")
        activate_relay(6, 1.0, RELAY_ACTIVE_LOW)
        time.sleep(1.0)
        
        print("\nTesting PIN 13 as RIGHT:")
        activate_relay(13, 1.0, RELAY_ACTIVE_LOW)
        time.sleep(1.0)
        
        print("\nTesting PIN 13 as LEFT:")
        activate_relay(13, 1.0, RELAY_ACTIVE_LOW)
        time.sleep(1.0)
        
        print("\nTesting PIN 6 as RIGHT:")
        activate_relay(6, 1.0, RELAY_ACTIVE_LOW)
        time.sleep(1.0)
        
        # Test for active high vs active low
        print("\nTesting PIN 6 with active HIGH:")
        GPIO.output(6, GPIO.HIGH)
        time.sleep(1.0)
        GPIO.output(6, GPIO.LOW)
        time.sleep(1.0)
        
        print("\nTesting PIN 6 with active LOW:")
        GPIO.output(6, GPIO.LOW)
        time.sleep(1.0)
        GPIO.output(6, GPIO.HIGH)
        time.sleep(1.0)
        
        print("\nTesting PIN 13 with active HIGH:")
        GPIO.output(13, GPIO.HIGH)
        time.sleep(1.0)
        GPIO.output(13, GPIO.LOW)
        time.sleep(1.0)
        
        print("\nTesting PIN 13 with active LOW:")
        GPIO.output(13, GPIO.LOW)
        time.sleep(1.0)
        GPIO.output(13, GPIO.HIGH)
        
    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"Error during automatic test: {e}")
    finally:
        cleanup()

def main():
    """Main function."""
    print("=== Relay Testing Tool ===")
    print("This tool helps diagnose which GPIO pins control which movement directions.")
    
    # Show valid pins
    print(f"Valid GPIO pins for testing: {VALID_PINS}")
    
    try:
        if len(sys.argv) > 1 and sys.argv[1] == 'auto':
            automatic_test()
        else:
            interactive_test()
    
        print("\nTest completed. Don't forget to update your main script with the correct pin assignments.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            cleanup()
        except:
            print("Note: Failed to clean up GPIO resources properly.") 