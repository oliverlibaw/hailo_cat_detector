#!/usr/bin/env python3
"""
Diagnostic script for the squirt relay issue.
Tests different approaches including pull-up/pull-down resistors.
"""

import RPi.GPIO as GPIO
import time
import sys

# SQUIRT relay pin
SQUIRT_PIN = 5

def setup():
    """Initialize GPIO with different options"""
    print("Setting up GPIO...")
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)  # Suppress warnings
    
    # Let's not set any initial output state yet
    print(f"Setting up pin {SQUIRT_PIN}...")
    GPIO.setup(SQUIRT_PIN, GPIO.OUT)
    
    # Start with a known state (OFF)
    print("Setting initial state to HIGH (assuming HIGH = OFF)...")
    GPIO.output(SQUIRT_PIN, GPIO.HIGH)

def cleanup():
    """Clean up GPIO resources"""
    print("Cleaning up GPIO...")
    # Try to set to HIGH before cleanup (assuming HIGH = OFF)
    try:
        GPIO.output(SQUIRT_PIN, GPIO.HIGH)
        print(f"Pin {SQUIRT_PIN} set to HIGH")
        time.sleep(0.5)  # Give it time to take effect
    except:
        pass
    GPIO.cleanup()
    print("GPIO cleanup completed")

def main():
    try:
        setup()
        
        print("\n=== SQUIRT RELAY DIAGNOSIS ===")
        print("This script will help diagnose issues with the squirt relay")
        
        # First, check if the relay is already activated
        print("\nIs the squirt relay currently ON? (y/n)")
        is_on = input("> ").lower().strip() == 'y'
        
        if is_on:
            print("Let's try to turn it OFF...")
            print("Setting pin to LOW...")
            GPIO.output(SQUIRT_PIN, GPIO.LOW)
            time.sleep(1)
            
            print("\nIs the relay OFF now? (y/n)")
            is_off = input("> ").lower().strip() == 'y'
            
            if is_off:
                print("Good! For this relay, LOW = OFF")
                off_state = GPIO.LOW
                on_state = GPIO.HIGH
            else:
                print("Let's try HIGH instead...")
                GPIO.output(SQUIRT_PIN, GPIO.HIGH)
                time.sleep(1)
                
                print("\nIs the relay OFF now? (y/n)")
                is_off = input("> ").lower().strip() == 'y'
                
                if is_off:
                    print("Good! For this relay, HIGH = OFF")
                    off_state = GPIO.HIGH
                    on_state = GPIO.LOW
                else:
                    print("Still not off? Let's try different pull-up/down resistors...")
                    # Try with pull-up resistor
                    GPIO.cleanup()
                    GPIO.setmode(GPIO.BCM)
                    GPIO.setup(SQUIRT_PIN, GPIO.OUT, pull_up_down=GPIO.PUD_UP)
                    GPIO.output(SQUIRT_PIN, GPIO.HIGH)
                    
                    print("\nIs the relay OFF now with pull-up resistor? (y/n)")
                    is_off = input("> ").lower().strip() == 'y'
                    
                    if is_off:
                        print("Success with pull-up resistor!")
                        off_state = GPIO.HIGH
                        on_state = GPIO.LOW
                    else:
                        # Try with pull-down resistor
                        GPIO.cleanup()
                        GPIO.setmode(GPIO.BCM)
                        GPIO.setup(SQUIRT_PIN, GPIO.OUT, pull_up_down=GPIO.PUD_DOWN)
                        GPIO.output(SQUIRT_PIN, GPIO.LOW)
                        
                        print("\nIs the relay OFF now with pull-down resistor? (y/n)")
                        is_off = input("> ").lower().strip() == 'y'
                        
                        if is_off:
                            print("Success with pull-down resistor!")
                            off_state = GPIO.LOW
                            on_state = GPIO.HIGH
                        else:
                            print("We're having trouble controlling this relay.")
                            print("Let's try a test loop to find what works...")
                            
                            # Just set a default to continue testing
                            off_state = GPIO.HIGH
                            on_state = GPIO.LOW
        else:
            print("Good! Let's determine what state keeps it OFF...")
            current_state = GPIO.HIGH  # Assuming we initialized to HIGH
            print(f"Current pin state is {'HIGH' if current_state == GPIO.HIGH else 'LOW'}")
            print("Let's remember this: DEFAULT=OFF state")
            
            # Set initial states based on current state
            off_state = current_state
            on_state = GPIO.LOW if current_state == GPIO.HIGH else GPIO.HIGH
        
        # Now test toggling
        print("\nLet's test toggling the relay on and off...")
        
        while True:
            print("\nCommands:")
            print("  1 - Turn relay ON")
            print("  0 - Turn relay OFF")
            print("  t - Toggle between ON and OFF repeatedly")
            print("  s - Switch ON/OFF state definitions")
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
            
            elif cmd == 't':
                count = int(input("Number of toggles: ") or "5")
                delay = float(input("Delay between toggles (seconds): ") or "1.0")
                
                for i in range(count):
                    print(f"Toggle {i+1}/{count}: ON")
                    GPIO.output(SQUIRT_PIN, on_state)
                    time.sleep(delay)
                    print(f"Toggle {i+1}/{count}: OFF")
                    GPIO.output(SQUIRT_PIN, off_state)
                    time.sleep(delay)
            
            elif cmd == 's':
                # Switch ON/OFF state definitions
                temp = on_state
                on_state = off_state
                off_state = temp
                print(f"Switched definitions: ON={'LOW' if on_state == GPIO.LOW else 'HIGH'}, OFF={'LOW' if off_state == GPIO.LOW else 'HIGH'}")
            
            elif cmd == 'q':
                # Make sure to turn off before quitting
                GPIO.output(SQUIRT_PIN, off_state)
                print("Turning relay OFF and quitting...")
                break
            
            else:
                print("Unknown command")
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        cleanup()

if __name__ == "__main__":
    main() 