#!/usr/bin/env python3
"""
PyroGuard AI Demo - Basic Wildfire Environment
Run: python demo.py
"""

from wildfire_env import WildfireEnvironment

def main():
    """Run the basic wildfire environment demo."""
    print("🔥 Starting PyroGuard Basic Demo...")
    
    # Create realistic large-scale wildfire environment with AI drones
    env = WildfireEnvironment(
        grid_size=30,            # Optimal size for drone operations
        fire_spread_prob=0.1,    # Much slower fire spread
        initial_tree_density=0.75,  # Not used anymore - realistic terrain generation
        wind_strength=0.05,      # Reduced wind effect
        fire_persistence=8,      # Fires burn longer but spread slower
        new_fire_rate=0.01,      # Very low automatic fire rate
        enable_drones=True       # Enable single AI firefighter drone
    )
    
    # Start the simulation
    state = env.reset()
    print(f"🌲 Realistic Forest Environment Initialized:")
    print(f"  - Total Vegetation: {state['total_vegetation']} units")
    print(f"  - Dense Forest: {state['dense_forest']}")
    print(f"  - Regular Trees: {state['trees_remaining']}")
    print(f"  - Sparse Forest: {state['sparse_forest']}")
    print(f"  - Grassland: {state['grassland']}")
    print(f"  - Water Bodies: {state['water_bodies']}")
    print(f"  - Rocky Terrain: {state['rocky_terrain']}")
    
    if 'drone_status' in state:
        drone_status = state['drone_status']
        print(f"\n🚁 AI Firefighter Drone Deployed:")
        print(f"  - Position: {drone_status['position']}")
        print(f"  - Water Level: {drone_status['water_percentage']:.0f}%")
        print(f"  - Energy Level: {drone_status['energy_percentage']:.0f}%")
        print(f"  - Fires Extinguished: {drone_status['fires_extinguished']}")
        print(f"  - Known Fires: {drone_status['known_fires']}")
    print("🎮 Interactive Controls:")
    print("  - SPACEBAR: Spawn random fires")
    print("  - W: Change wind direction")
    print("  - R: Reset to fresh forest")
    print("  - CLICK: Ignite specific trees")
    print("  - Close window or Ctrl+C to stop")
    print("\n🌲 Forest is peaceful - click or press SPACE to start fires!")
    
    try:
        step = 0
        print("🎮 Demo running! Use controls to interact with the forest.")
        
        while True:  # Run forever until user closes window or Ctrl+C
            if not env.render():
                print("👋 Window closed by user")
                break
            
            state = env.step()
            step += 1
            
            # Print status updates less frequently and only when there's action
            if step % 50 == 0 and state['fires_active'] > 0:
                drone_info = ""
                if 'drone_status' in state:
                    ds = state['drone_status']
                    drone_info = f", Drone: {ds['fires_extinguished']} extinguished, {ds['known_fires']} known"
                
                print(f"Step {step}: {state['fires_active']} active fires "
                      f"({state['fires_intense']} intense), {state['trees_remaining']} trees left{drone_info}")
            
            elif step % 100 == 0 and state['fires_active'] == 0:
                drone_info = ""
                if 'drone_status' in state:
                    ds = state['drone_status']
                    drone_info = f" - Drone: {ds['fires_extinguished']} fires suppressed total"
                
                print(f"Step {step}: Forest peaceful - {state['total_vegetation']} vegetation units healthy{drone_info}")
        
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user (Ctrl+C)")
    finally:
        print("🔥 Thanks for using PyroGuard!")
        env.close()

if __name__ == "__main__":
    main()
