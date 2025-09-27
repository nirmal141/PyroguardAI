"""
Test script to verify fire dynamics are working correctly.
"""

import sys
import os
import numpy as np

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.fire_dynamics import TREE, BURNING, BURNED, COMPLETELY_BURNED, step_fire, count_states, ignite_random

def test_fire_dynamics():
    """Test the fire dynamics progression."""
    print("🔥 Testing Fire Dynamics")
    print("=" * 40)
    
    # Create a small test grid
    grid = np.full((5, 5), TREE, dtype=np.int8)
    
    # Ignite one fire in the center
    grid[2, 2] = BURNING
    
    print("Initial state:")
    print_grid(grid)
    
    # Run simulation for several steps
    for step in range(10):
        print(f"\nStep {step + 1}:")
        grid = step_fire(grid, p_spread=0.3, p_burnout=0.2)
        tree, burning, burned, completely_burned = count_states(grid)
        print(f"  Trees: {tree}, Burning: {burning}, Burned: {burned}, Completely Burnt: {completely_burned}")
        print_grid(grid)
        
        # Stop if no more burning cells
        if burning == 0:
            print("  No more burning cells!")
            break
    
    print("\n✅ Fire dynamics test completed!")

def print_grid(grid):
    """Print grid with symbols."""
    symbols = {TREE: "🌲", BURNING: "🔥", BURNED: "🍑", COMPLETELY_BURNED: "⬛"}
    
    for row in grid:
        print("".join(symbols[cell] for cell in row))

if __name__ == "__main__":
    test_fire_dynamics()
