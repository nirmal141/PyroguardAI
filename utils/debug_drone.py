#!/usr/bin/env python3
"""
Debug script to check drone initialization
"""

from demo_rl import RLIntegratedWildfireEnvironment

def test_drone_initialization():
    print("Testing drone initialization...")
    
    # Test simple drone
    env = RLIntegratedWildfireEnvironment(grid_size=15, drone_type='simple')
    print(f"Drone type: {env.drone_type}")
    print(f"Has drone attribute: {hasattr(env, 'drone')}")
    
    if hasattr(env, 'drone') and env.drone:
        print(f"Drone position: {env.drone.position}")
        print(f"Drone water: {env.drone.water_level}")
        print(f"Drone energy: {env.drone.energy}")
    else:
        print("❌ No drone found!")
    
    # Test a few simulation steps
    print("\nTesting simulation steps...")
    state = env.reset()
    
    for i in range(3):
        prev_state = state
        state = env.step()
        
        drone_action = state.get('drone_action', {})
        drone_status = state.get('drone_status', {})
        
        print(f"Step {i+1}:")
        print(f"  Drone action: {drone_action.get('action', 'none')}")
        print(f"  Drone position: {drone_status.get('position', 'unknown')}")
    
    env.close()
    print("✅ Test completed")

if __name__ == "__main__":
    test_drone_initialization()