
import numpy as np
import sys
import os

# Ensure we can import modules from the current directory
sys.path.append(os.getcwd())

from system_configuration import Config
from com_components import User

def verify_angle():
    cfg = Config()
    # Mock antenna array (needed for User init but not used for this test)
    antenna_array = np.zeros(cfg.N_antennaElement)
    
    user = User(antenna_array, cfg)
    
    # Check 1: Verify angle logic
    # Expected LOS angle
    los_angle = np.arctan2(user.y, user.x)
    
    # Verify that angle_of_direction is centered around los_angle
    # The spread is random, so we can't check exact values, but we can check if it's within expected range
    # 5 degrees spread is relatively small.
    # We can perform a statistical check or just print for manual review.
    
    print(f"User Position: ({user.x}, {user.y})")
    print(f"LOS Angle (rad): {los_angle}")
    print(f"Multipath Spread Config (rad): {cfg.multipath_angle_spread}")
    print("Generated Angles of Direction:")
    print(user.angle_of_direction)
    
    diffs = user.angle_of_direction - los_angle
    print("Differences from LOS:")
    print(diffs)
    
    # Check if all diffs are somewhat reasonable (e.g., within 3-4 std devs)
    # 5 degrees is ~0.087 rad
    std_dev = cfg.multipath_angle_spread
    max_diff = np.max(np.abs(diffs))
    
    print(f"Max deviation: {max_diff} rad ({max_diff * 180 / np.pi} degrees)")
    
    if max_diff < 5 * std_dev: # 5 sigma check
        print("PASS: Angles are within reasonable spread.")
    else:
        print("FAIL: Angles deviate too much.")

if __name__ == "__main__":
    verify_angle()
