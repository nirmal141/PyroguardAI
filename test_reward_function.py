#!/usr/bin/env python3
"""
Test the improved reward function to verify it's working correctly.
"""

import numpy as np
from rl_wildfire_env import RLWildfireEnv

def test_reward_function():
    """Test the improved reward function."""
    print("🧪 Testing Improved Reward Function")
    print("=" * 50)
    
    # Create environment
    env = RLWildfireEnv(grid_size=25, max_steps=50)
    
    # Test multiple episodes
    total_rewards = []
    
    for episode in range(3):
        print(f"\nEpisode {episode + 1}:")
        obs, _ = env.reset()
        episode_reward = 0
        
        for step in range(20):  # Test first 20 steps
            # Take random action
            action = np.random.randint(0, 6)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            
            if step < 5:  # Print first few steps
                print(f"  Step {step+1}: Action={action}, Reward={reward:.2f}, Fires={info['fires_active']}")
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        print(f"  Episode total reward: {episode_reward:.2f}")
    
    # Analyze results
    avg_reward = np.mean(total_rewards)
    print(f"\n📊 Results:")
    print(f"Average reward per episode: {avg_reward:.2f}")
    print(f"Individual episode rewards: {[f'{r:.2f}' for r in total_rewards]}")
    
    if avg_reward > 0:
        print("✅ Reward function is now positive!")
    elif avg_reward > -5:
        print("⚠️  Reward function is much better (less negative)")
    else:
        print("❌ Reward function still needs improvement")

if __name__ == "__main__":
    test_reward_function()
