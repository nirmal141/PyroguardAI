"""
Demo script for multi-agent drone firefighting simulation.
Shows how to use the system with manual controls and RL training.
"""

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.multi_agent_fire_env import MultiAgentFireEnv
from envs.drone_agent import DroneAction
import numpy as np

def demo_manual_control():
    """Demo with manual control of drones."""
    print("🔥 Multi-Agent Drone Firefighting Demo")
    print("=" * 50)
    
    # Create environment
    env = MultiAgentFireEnv(
        width=12, 
        height=12, 
        p_spread=0.3, 
        p_burnout=0.1, 
        max_steps=500, 
        ignitions=(1, 2),
        num_drones=2,
        seed=42
    )
    
    # Reset environment
    obs, _ = env.reset()
    print(f"Environment initialized with {env.num_drones} drones")
    print(f"Grid size: {env.width}x{env.height}")
    
    # Activate firefighting mode
    env.activate_drone_firefighting()
    print("🚁 Drone firefighting activated!")
    
    # Run simulation
    done = False
    step = 0
    
    while not done and step < 100:
        # Simple heuristic actions
        actions = []
        for i in range(env.num_drones):
            drone_obs = obs[i]
            x, y, battery, water, temperature, fire_detected = drone_obs[:6]
            
            # Simple strategy: if fire detected and have water, extinguish
            if fire_detected > 0.5 and water > 0.1:
                actions.append(DroneAction.EXTINGUISH.value)
            elif temperature > 0.3:
                # Move towards fire (random direction for demo)
                actions.append(np.random.randint(0, 4))
            else:
                # Scan temperature
                actions.append(DroneAction.SCAN_TEMPERATURE.value)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(np.array(actions))
        step += 1
        
        # Print status
        if step % 10 == 0:
            print(f"Step {step}: Fires={info['burning']}, "
                  f"Active Drones={info['drones_active']}, "
                  f"Extinguished={info['fires_extinguished']}")
        
        # Check termination
        done = terminated or truncated
    
    print(f"\n🎯 Demo completed!")
    print(f"  Steps: {step}")
    print(f"  Fires extinguished: {env.total_fires_extinguished}")
    print(f"  Final fires: {info['burning']}")

def demo_drone_capabilities():
    """Demo showing drone capabilities and sensors."""
    print("\n🚁 Drone Capabilities Demo")
    print("=" * 30)
    
    # Create environment
    env = MultiAgentFireEnv(
        width=8, 
        height=8, 
        num_drones=1,
        seed=123
    )
    
    obs, _ = env.reset()
    drone = env.drones[0]
    
    print(f"Initial drone state:")
    print(f"  Position: ({drone.state.x}, {drone.state.y})")
    print(f"  Battery: {drone.state.battery:.2f}")
    print(f"  Water: {drone.state.water_capacity:.2f}")
    print(f"  Temperature: {drone.state.temperature_reading:.2f}")
    
    # Activate firefighting
    env.activate_drone_firefighting()
    
    # Test different actions
    print(f"\nTesting drone actions:")
    
    # Scan temperature
    result = drone.execute_action(DroneAction.SCAN_TEMPERATURE, env.grid, env.width, env.height)
    print(f"  Scan temperature: {result['message']}")
    
    # Move around
    for action in [DroneAction.MOVE_RIGHT, DroneAction.MOVE_DOWN, DroneAction.MOVE_LEFT, DroneAction.MOVE_UP]:
        result = drone.execute_action(action, env.grid, env.width, env.height)
        print(f"  Move {action.name}: Position now ({drone.state.x}, {drone.state.y})")
    
    # Check final state
    print(f"\nFinal drone state:")
    print(f"  Position: ({drone.state.x}, {drone.state.y})")
    print(f"  Battery: {drone.state.battery:.2f}")
    print(f"  Water: {drone.state.water_capacity:.2f}")
    print(f"  Temperature: {drone.state.temperature_reading:.2f}")

if __name__ == "__main__":
    print("🚁 Multi-Agent Drone Firefighting System Demo")
    print("=" * 60)
    
    # Demo 1: Manual control
    demo_manual_control()
    
    # Demo 2: Drone capabilities
    demo_drone_capabilities()
    
    print("\n✅ Demo completed!")
    print("\nTo run the full simulation with controls:")
    print("  python scripts/run_multi_agent_fire_sim.py")
    print("\nTo train RL agents:")
    print("  python scripts/train_drone_rl.py")
    print("\nTo test trained agents:")
    print("  python scripts/train_drone_rl.py test")
