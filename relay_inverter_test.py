#!/usr/bin/env python3
"""
Script to test if the relay logic needs to be inverted.
This script will try different combinations of RELAY_ACTIVE_LOW and RELAY_NORMALLY_CLOSED 
to find the correct logic for your specific relay hardware.
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

# Timing configuration
ACTIVATION_DURATION = 3.0  # Seconds to keep each relay active
PAUSE_DURATION = 1.0       # Seconds to pause between tests

def setup_gpio():
    """Initialize GPIO pins for relays"""
    GPIO.setmode(GPIO.BCM)
    for pin in RELAY_PINS.values():
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.HIGH)  # Start with all pins HIGH
    print("GPIO pins initialized")

def cleanup_gpio():
    """Clean up GPIO resources"""
    print("\nCleaning up GPIO resources...")
    GPIO.cleanup()
    print("GPIO cleanup completed")

def test_relay_with_settings(pin, name, active_low, normally_closed):
    """Test a relay with specific settings for ACTIVE_LOW and NORMALLY_CLOSED"""
    
    print(f"\nTesting {name.upper()} relay (pin {pin}) with:")
    print(f"  ACTIVE_LOW = {active_low}")
    print(f"  NORMALLY_CLOSED = {normally_closed}")
    
    # Calculate the GPIO state to activate the relay
    # If active_low and normally_closed have the same value (both True or both False),
    # setting the relay to "ON" requires a LOW GPIO signal
    on_state = GPIO.LOW if active_low == normally_closed else GPIO.HIGH
    off_state = GPIO.HIGH if on_state == GPIO.LOW else GPIO.LOW
    
    print(f"ON state requires GPIO {'LOW' if on_state == GPIO.LOW else 'HIGH'}")
    print(f"OFF state requires GPIO {'LOW' if off_state == GPIO.LOW else 'HIGH'}")
    
    # First ensure the relay is OFF
    print(f"Setting relay to OFF state (GPIO {'LOW' if off_state == GPIO.LOW else 'HIGH'})...")
    GPIO.output(pin, off_state)
    time.sleep(PAUSE_DURATION)
    
    # Then turn it ON
    print(f"Setting relay to ON state (GPIO {'LOW' if on_state == GPIO.LOW else 'HIGH'})...")
    print(f"Relay should be ACTIVE now - Does the motor move? (waiting {ACTIVATION_DURATION} seconds)")
    GPIO.output(pin, on_state)
    time.sleep(ACTIVATION_DURATION)
    
    # Turn it OFF again
    print("Setting relay back to OFF state...")
    GPIO.output(pin, off_state)
    
    # Ask for user feedback
    result = input("Did the motor move properly? (y/n): ").strip().lower()
    
    return result == 'y'

def test_all_combinations():
    """Test all combinations of ACTIVE_LOW and NORMALLY_CLOSED for each relay"""
    results = {}
    
    for name, pin in RELAY_PINS.items():
        if name == 'unused':
            continue
            
        print(f"\n{'='*50}")
        print(f"TESTING {name.upper()} RELAY (PIN {pin})")
        print(f"{'='*50}")
        
        results[name] = []
        
        # Test all four combinations
        for active_low in [True, False]:
            for normally_closed in [True, False]:
                success = test_relay_with_settings(pin, name, active_low, normally_closed)
                if success:
                    results[name].append((active_low, normally_closed))
    
    return results

def main():
    """Main function"""
    try:
        setup_gpio()
        
        print("\n===== RELAY INVERTER TEST =====")
        print("This script will test different relay logic configurations")
        print("to find the correct settings for your hardware.")
        print("\nFor each test, watch if the motor moves correctly.")
        print("Answer 'y' if it works as expected, 'n' if not.")
        print("There will be 4 tests per relay (12 tests total).")
        print("\nReady to begin? Press Enter to continue or Ctrl+C to exit.")
        input()
        
        results = test_all_combinations()
        
        # Print summary of results
        print("\n===== TEST RESULTS =====")
        
        valid_configs_found = False
        
        for name, configs in results.items():
            print(f"\n{name.upper()} relay (pin {RELAY_PINS[name]}):")
            if configs:
                valid_configs_found = True
                for active_low, normally_closed in configs:
                    print(f"  SUCCESS with ACTIVE_LOW={active_low}, NORMALLY_CLOSED={normally_closed}")
            else:
                print("  No successful configurations found!")
        
        if valid_configs_found:
            print("\nBased on your results, update your main.py and test_relays.py with:")
            
            # Try to find a common configuration that works for all relays
            common_configs = []
            for active_low in [True, False]:
                for normally_closed in [True, False]:
                    if all((active_low, normally_closed) in configs for configs in results.values() if configs):
                        common_configs.append((active_low, normally_closed))
            
            if common_configs:
                active_low, normally_closed = common_configs[0]
                print(f"\nRELAY_ACTIVE_LOW = {active_low}")
                print(f"RELAY_NORMALLY_CLOSED = {normally_closed}")
            else:
                print("\nNo common configuration works for all relays!")
                print("You may need different handling for each relay.")
        else:
            print("\nNo successful configurations found for any relay!")
            print("Check your wiring and connections.")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        cleanup_gpio()
        print("\nTest completed")

if __name__ == "__main__":
    main() 