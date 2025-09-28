#!/usr/bin/env python3
"""
Debug script to test drone visibility in original vs RL demo
"""

def test_original_demo():
    print("🧪 Testing Original Demo...")
    from wildfire_env import WildfireEnvironment
    
    env = WildfireEnvironment(
        grid_size=15,
        enable_drones=True
    )
    
    state = env.reset()
    print(f"Enable drones: {env.enable_drones}")
    print(f"Has drone: {hasattr(env, 'drone') and env.drone is not None}")
    
    if hasattr(env, 'drone') and env.drone:
        print(f"Drone position: {env.drone.position}")
        print(f"Drone water: {env.drone.water_level}")
        
    print(f"State has drone_status: {'drone_status' in state}")
    if 'drone_status' in state:
        print(f"Drone status: {state['drone_status']}")
    
    env.close()
    print("✅ Original demo test complete\n")

def test_rl_demo():
    print("🧪 Testing RL Demo...")
    from demo_rl import RLIntegratedWildfireEnvironment
    
    env = RLIntegratedWildfireEnvironment(
        grid_size=15,
        drone_type='simple'
    )
    
    state = env.reset()
    print(f"Enable drones: {env.enable_drones}")
    print(f"Has drone: {hasattr(env, 'drone') and env.drone is not None}")
    
    if hasattr(env, 'drone') and env.drone:
        print(f"Drone position: {env.drone.position}")
        print(f"Drone water: {env.drone.water_level}")
    
    print(f"State has drone_status: {'drone_status' in state}")
    if 'drone_status' in state:
        print(f"Drone status: {state['drone_status']}")
    
    env.close()
    print("✅ RL demo test complete\n")

if __name__ == "__main__":
    test_original_demo()
    test_rl_demo()