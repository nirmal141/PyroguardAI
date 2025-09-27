import pygame
import time
import sys
import os

# Add the parent directory to the path so we can import from envs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.wildfire_env import FireSimEnv

def main():
    env = FireSimEnv(width=16, height=16, p_spread=0.4, p_burnout=0.2, max_steps=1000, ignitions=(1,2), seed=42)
    obs, _ = env.reset()
    done = False
    
    # Initialize pygame display by calling render once
    env.render(mode="human")
    
    try:
        while not done:
            # Handle pygame events to keep window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break
            
            if not done:
                obs, reward, term, trunc, info = env.step()
                env.render(mode="human")
                print(f"step={info['steps']} burning={info['burning']} reward={reward}")
                print("-"*24)
                # Stop when fire naturally ends (no more burning cells) or when manually closed
                done = term
                
                # Add delay to make animation visible
                time.sleep(0.5)
        
        # Simulation completed - keep window open until user closes it
        print("\nSimulation completed! Close the pygame window to exit.")
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
            time.sleep(0.1)  # Small delay to prevent high CPU usage
    
    finally:
        # Clean up pygame
        pygame.quit()

if __name__ == "__main__":
    main()
