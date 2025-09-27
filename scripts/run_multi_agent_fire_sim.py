"""
Multi-agent firefighting simulation with drone control.
Features trigger-based activation and RL training integration.
"""

import pygame
import time
import sys
import os
import numpy as np
from typing import Optional

# Add the parent directory to the path so we can import from envs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.multi_agent_fire_env import MultiAgentFireEnv
from envs.drone_agent import DroneAction

class DroneController:
    """Simple controller for drone actions (can be replaced with RL agent)."""
    
    def __init__(self, num_drones: int):
        self.num_drones = num_drones
        self.action_history = []
    
    def get_actions(self, observations: np.ndarray, env: MultiAgentFireEnv) -> np.ndarray:
        """
        Get actions for all drones based on observations.
        This is a simple heuristic controller - replace with RL agent.
        """
        actions = np.zeros(self.num_drones, dtype=int)
        
        for i in range(self.num_drones):
            drone_obs = observations[i]
            x, y, battery, water, temperature, fire_detected = drone_obs[:6]
            
            # Simple heuristic strategy
            if fire_detected > 0.5 and water > 0.1:
                # Fire detected and have water - extinguish
                actions[i] = DroneAction.EXTINGUISH.value
            elif temperature > 0.3:
                # High temperature nearby - move towards fire
                # Random movement including diagonal
                actions[i] = np.random.randint(0, 8)  # 8 movement directions
            elif battery < 0.3:
                # Low battery - try to stay in place
                actions[i] = DroneAction.STAY.value
            else:
                # Default: scan temperature to gather information
                actions[i] = DroneAction.SCAN_TEMPERATURE.value
        
        return actions

def main():
    """Main simulation loop with interactive controls."""
    # Initialize environment
    env = MultiAgentFireEnv(
        width=16, 
        height=16, 
        p_spread=0.4, 
        p_burnout=0.15, 
        max_steps=1000, 
        ignitions=(2, 4),
        num_drones=3,
        seed=42
    )
    
    # Initialize controller
    controller = DroneController(env.num_drones)
    
    # Reset environment
    obs, _ = env.reset()
    done = False
    
    # Initialize pygame display
    env.render(mode="human")
    
    # Control variables
    running = True
    paused = True  # Start paused
    auto_mode = False
    step_count = 0
    
    # Pause at start to let user see initial state
    print("⏸️  Simulation paused at start. Press SPACE to begin!")
    
    print("🔥 Multi-Agent Firefighting Simulation")
    print("=" * 50)
    print("🎮 CONTROLS:")
    print("  SPACE - Toggle pause/resume")
    print("  T - 🚁 ACTIVATE DRONES (trigger button)")
    print("  R - 🔋 Recharge drones")
    print("  A - Toggle auto mode")
    print("  ESC - Exit")
    print("=" * 50)
    print("💡 TIP: Press 'T' to activate drone firefighting!")
    print("💡 TIP: Press SPACE to pause and take your time!")
    print("=" * 50)
    
    try:
        while running and not done:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print(f"⏸️  {'Paused' if paused else 'Resumed'}")
                    elif event.key == pygame.K_t:
                        if env.drone_firefighting_active:
                            env.deactivate_drone_firefighting()
                        else:
                            env.activate_drone_firefighting()
                    elif event.key == pygame.K_r:
                        env.recharge_drones()
                    elif event.key == pygame.K_a:
                        auto_mode = not auto_mode
                        print(f"🤖 Auto mode: {'ON' if auto_mode else 'OFF'}")
            
            if not running:
                break
                
            if not paused:
                # Get actions from controller
                if env.drone_firefighting_active:
                    actions = controller.get_actions(obs, env)
                else:
                    # No actions when firefighting is inactive
                    actions = np.array([DroneAction.STAY.value] * env.num_drones)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(actions)
                step_count += 1
                
                # Print status every 10 steps
                if step_count % 10 == 0:
                    print(f"Step {step_count}: Fires={info['burning']}, "
                          f"Active Drones={info['drones_active']}, "
                          f"Extinguished={info['fires_extinguished']}, "
                          f"Reward={reward:.2f}")
                
                # Check termination
                done = terminated or truncated
                
                # Auto mode: automatically activate firefighting when fires appear
                if auto_mode and not env.drone_firefighting_active and info['burning'] > 0:
                    env.activate_drone_firefighting()
                
                # Render
                env.render(mode="human")
                
                # Small delay for visualization (drones move faster than fire)
                time.sleep(0.2)  # Faster drone movement
            else:
                # Still render when paused
                env.render(mode="human")
                time.sleep(0.1)
        
        # Simulation completed
        print("\n🎯 Simulation Results:")
        print(f"  Total steps: {step_count}")
        print(f"  Fires extinguished: {env.total_fires_extinguished}")
        print(f"  Final fires remaining: {info['burning']}")
        
        # Keep window open until user closes it
        print("\nClose the pygame window to exit.")
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Simulation interrupted by user")
    
    finally:
        # Clean up
        pygame.quit()

if __name__ == "__main__":
    main()
