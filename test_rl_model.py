#!/usr/bin/env python3
"""
Test script to verify RL model is making decisions different from random.
"""

import numpy as np
import random
from stable_baselines3 import PPO
from rl_wildfire_env import RLWildfireEnv

def test_rl_vs_random():
    """Test if RL model makes different decisions than random."""
    print("🧪 Testing RL Model vs Random Decisions")
    print("=" * 50)
    
    # Load RL model
    try:
        rl_model = PPO.load("models/final_model_ppo.zip")
        print("✅ RL model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading RL model: {e}")
        return
    
    # Create environment
    env = RLWildfireEnv(grid_size=25, max_steps=50)
    
    # Test multiple episodes
    rl_actions = []
    random_actions = []
    
    for episode in range(5):
        print(f"\nEpisode {episode + 1}:")
        obs, _ = env.reset()
        
        rl_episode_actions = []
        random_episode_actions = []
        
        for step in range(20):  # Test first 20 steps
            # Get RL action
            rl_action, _ = rl_model.predict(obs, deterministic=True)
            rl_episode_actions.append(rl_action)
            
            # Get random action
            random_action = random.randint(0, 5)
            random_episode_actions.append(random_action)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(rl_action)
            
            if terminated or truncated:
                break
        
        rl_actions.extend(rl_episode_actions)
        random_actions.extend(random_episode_actions)
        
        print(f"  RL actions: {rl_episode_actions[:10]}...")
        print(f"  Random actions: {random_episode_actions[:10]}...")
    
    # Analyze action distributions
    rl_action_counts = np.bincount(rl_actions, minlength=6)
    random_action_counts = np.bincount(random_actions, minlength=6)
    
    print(f"\n📊 Action Distribution Analysis:")
    print(f"{'Action':<10} {'RL Count':<10} {'Random Count':<12} {'Difference':<12}")
    print("-" * 50)
    
    for action in range(6):
        rl_count = rl_action_counts[action]
        random_count = random_action_counts[action]
        difference = rl_count - random_count
        print(f"{action:<10} {rl_count:<10} {random_count:<12} {difference:<12}")
    
    # Calculate similarity
    total_actions = len(rl_actions)
    same_actions = sum(1 for rl, rand in zip(rl_actions, random_actions) if rl == rand)
    similarity = same_actions / total_actions * 100
    
    print(f"\n🎯 Analysis:")
    print(f"Total actions: {total_actions}")
    print(f"Same actions: {same_actions}")
    print(f"Similarity: {similarity:.1f}%")
    
    if similarity > 80:
        print("❌ RL model is behaving very similarly to random!")
    elif similarity > 60:
        print("⚠️  RL model shows some differences but still quite random")
    elif similarity > 40:
        print("✅ RL model shows moderate differences from random")
    else:
        print("🎉 RL model shows significant differences from random")
    
    # Test reward function
    print(f"\n🔥 Testing Reward Function:")
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)  # Move North
    print(f"Step 1 - Reward: {reward:.2f}, Info: {info}")
    
    obs, reward, terminated, truncated, info = env.step(5)  # Suppress
    print(f"Step 2 - Reward: {reward:.2f}, Info: {info}")

if __name__ == "__main__":
    test_rl_vs_random()
