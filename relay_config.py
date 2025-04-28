#!/usr/bin/env python3
"""
Configuration file for relay pins and behavior.
This file allows for easy reconfiguration without modifying the main application code.
"""

# Relay pin assignments - Change these based on test results
RELAY_PINS = {
    'squirt': 5,     # Squirt relay (triggers water gun)
    'left': 6,       # LEFT movement relay (move camera/gun to the left)
    'right': 13,     # RIGHT movement relay (move camera/gun to the right)
    'unused': 15     # Unused relay 
}

# Relay behavior settings
RELAY_ACTIVE_LOW = True  # Set to True if relays activate on LOW signal (most HATs)

# Squirt relay behavior settings (may be different from movement relays)
SQUIRT_RELAY_ON_STATE = 1    # GPIO.HIGH (1) to activate
SQUIRT_RELAY_OFF_STATE = 0   # GPIO.LOW (0) to deactivate

# PD Control Parameters (Adjust after testing)
PD_CENTER_THRESHOLD = 0.10   # Dead zone around center (+/- %) - INCREASED for stability
PD_KP = 0.60                 # Proportional gain
PD_KD = 0.10                 # Derivative gain
PD_MIN_PULSE = 0.01          # Minimum effective pulse duration (seconds)
PD_MAX_PULSE = 0.15          # Maximum pulse duration (seconds)
PD_MOVEMENT_COOLDOWN = 0.2   # Cooldown between movements (seconds) - INCREASED for stability

# Verification flag - Set this to True after verifying the configuration is correct
CONFIGURATION_VERIFIED = False

# Configuration notes (for documentation)
CONFIGURATION_NOTES = """
Important configuration notes:
1. The 'left' relay should move the camera/gun to the LEFT
2. The 'right' relay should move the camera/gun to the RIGHT
3. If movements are reversed, swap the pin numbers for 'left' and 'right'
4. Most relay HATs are active-low (relay activates when GPIO pin is set LOW)
5. The squirt relay may have different behavior than the movement relays
6. Higher PD_CENTER_THRESHOLD gives more stability but less accuracy
7. Higher PD_MOVEMENT_COOLDOWN prevents oscillation but slows response
8. Set CONFIGURATION_VERIFIED to True once testing confirms correct operation
"""

# Function to print configuration details for verification
def print_config():
    """Print the current configuration settings for verification."""
    print("\n=== RELAY CONFIGURATION ===")
    print(f"SQUIRT relay: PIN {RELAY_PINS['squirt']}")
    print(f"LEFT movement relay: PIN {RELAY_PINS['left']} (moves camera/gun LEFT)")
    print(f"RIGHT movement relay: PIN {RELAY_PINS['right']} (moves camera/gun RIGHT)")
    print(f"Relays are {'ACTIVE LOW' if RELAY_ACTIVE_LOW else 'ACTIVE HIGH'}")
    print(f"Squirt relay: ON={SQUIRT_RELAY_ON_STATE}, OFF={SQUIRT_RELAY_OFF_STATE}")
    print("\n=== PD CONTROL PARAMETERS ===")
    print(f"Center threshold: ±{PD_CENTER_THRESHOLD} (dead zone)")
    print(f"KP: {PD_KP} (proportional gain)")
    print(f"KD: {PD_KD} (derivative gain)")
    print(f"Movement cooldown: {PD_MOVEMENT_COOLDOWN}s")
    print(f"Configuration verified: {'YES' if CONFIGURATION_VERIFIED else 'NO - TESTING NEEDED'}")
    print("===========================\n")

if __name__ == "__main__":
    # If this file is run directly, print configuration
    print_config()
    
    if not CONFIGURATION_VERIFIED:
        print("WARNING: Configuration has not been verified!")
        print("Please run test_relays.py to determine the correct settings.") 