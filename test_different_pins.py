#!/usr/bin/env python3
"""
Script to test different GPIO pins to see if the issue is pin-specific.
"""

import RPi.GPIO as GPIO
import time
import sys

# Try different possible pins
TEST_PINS = [5, 6, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

def setup():
    """Initialize GPIO"""
    GPIO.setmode(GPIO.BCM)
    for pin in TEST_PINS:
        try:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.HIGH)  # Start with all pins HIGH
            print(f"GPIO pin {pin} initialized to HIGH")
        except Exception as e:
            print(f"Could not initialize pin {pin}: {e}")

def cleanup():
    """Clean up GPIO resources"""
    print("Cleaning up GPIO...")
    GPIO.cleanup()
    print("GPIO cleanup completed")

def main():
    try:
        setup()
        
        print("\nAll available pins have been initialized to HIGH")
        print("Next, we'll cycle through different pins to test them")
        
        while True:
            print("\nOptions:")
            for i, pin in enumerate(TEST_PINS):
                print(f"  {i+1} - Test pin {pin}")
            print("  a - Set ALL pins HIGH")
            print("  z - Set ALL pins LOW")
            print("  q - Quit")
            
            cmd = input("\nEnter command: ").strip().lower()
            
            if cmd == 'q':
                break
            elif cmd == 'a':
                for pin in TEST_PINS:
                    try:
                        GPIO.output(pin, GPIO.HIGH)
                        print(f"Pin {pin} set to HIGH")
                    except Exception as e:
                        print(f"Error setting pin {pin}: {e}")
                print("All pins set to HIGH")
            elif cmd == 'z':
                for pin in TEST_PINS:
                    try:
                        GPIO.output(pin, GPIO.LOW)
                        print(f"Pin {pin} set to LOW")
                    except Exception as e:
                        print(f"Error setting pin {pin}: {e}")
                print("All pins set to LOW")
            elif cmd.isdigit() and 1 <= int(cmd) <= len(TEST_PINS):
                pin_index = int(cmd) - 1
                pin = TEST_PINS[pin_index]
                
                print(f"\nTesting pin {pin}:")
                print("  1 - Set to HIGH")
                print("  0 - Set to LOW")
                print("  b - Toggle repeatedly")
                print("  r - Return to main menu")
                
                while True:
                    subcmd = input("\nPin command: ").strip().lower()
                    
                    if subcmd == '1':
                        GPIO.output(pin, GPIO.HIGH)
                        print(f"Pin {pin} set to HIGH")
                    elif subcmd == '0':
                        GPIO.output(pin, GPIO.LOW)
                        print(f"Pin {pin} set to LOW")
                    elif subcmd == 'b':
                        count = int(input("Number of toggles: ") or "5")
                        delay = float(input("Delay between toggles (seconds): ") or "0.5")
                        
                        for i in range(count):
                            print(f"Toggle {i+1}/{count}: HIGH")
                            GPIO.output(pin, GPIO.HIGH)
                            time.sleep(delay)
                            print(f"Toggle {i+1}/{count}: LOW")
                            GPIO.output(pin, GPIO.LOW)
                            time.sleep(delay)
                        
                        # End with HIGH (assuming this is OFF for most relays)
                        GPIO.output(pin, GPIO.HIGH)
                        print(f"Pin {pin} returned to HIGH")
                    elif subcmd == 'r':
                        break
                    else:
                        print("Unknown command")
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