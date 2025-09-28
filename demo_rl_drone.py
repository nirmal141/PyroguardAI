#!/usr/bin/env python3
"""
Demo script for trained PyroGuard AI RL Drone
Shows the trained drone in action with visualization.
"""

import sys
import os
from stable_baselines3 import PPO
from rl_wildfire_env import RLWildfireEnv


def demo_trained_drone(model_path="models/final_model_ppo.zip", grid_size=35, max_steps=1000):
    """Demo the trained drone with visualization."""
    
    print("🚁 PyroGuard AI - Trained Drone Demo")
    print("=" * 40)
    
    # Load the trained model
    try:
        model = PPO.load(model_path)
        print(f"✅ Loaded trained model from {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create environment
    env = RLWildfireEnv(grid_size=grid_size, max_steps=max_steps)
    
    # Reset environment
    obs, _ = env.reset()
    
    print("🌲 Environment initialized!")
    print("🎮 Controls:")
    print("  - Watch the trained drone fight fires")
    print("  - Close window to stop demo")
    print("  - Press Ctrl+C to stop")
    
    total_reward = 0
    step = 0
    
    try:
        while True:
            # Render the environment
            if not env.render():
                break
            
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step += 1
            
            # Print periodic updates
            if step % 50 == 0:
                print(f"Step {step}: Reward {total_reward:.1f}, "
                      f"Fires: {info['fires_active']}, "
                      f"Extinguished: {info['fires_extinguished']}, "
                      f"Energy: {info['drone_energy']}")
            
            # Check termination
            if terminated:
                print(f"🎉 All fires extinguished! Total reward: {total_reward:.1f}")
                break
            elif truncated:
                print(f"⏰ Time limit reached! Total reward: {total_reward:.1f}")
                break
                
    except KeyboardInterrupt:
        print(f"\n👋 Demo stopped by user. Total reward: {total_reward:.1f}")
    except Exception as e:
        print(f"❌ Error during demo: {e}")
    finally:
        env.close()
        print("🔥 Thanks for watching PyroGuard AI!")


def compare_with_random():
    """Compare trained drone with random actions."""
    print("🔬 Comparing Trained vs Random Drone")
    print("=" * 40)
    
    # Test trained model
    print("🤖 Testing Trained Drone...")
    trained_rewards = test_drone_performance("models/final_model_ppo.zip", num_episodes=5)
    
    # Test random actions
    print("\n🎲 Testing Random Drone...")
    random_rewards = test_random_drone(num_episodes=5)
    
    # Compare results
    print(f"\n📊 Comparison Results:")
    print(f"  Trained Drone: {trained_rewards['avg_reward']:.1f} ± {trained_rewards['std_reward']:.1f}")
    print(f"  Random Drone:  {random_rewards['avg_reward']:.1f} ± {random_rewards['std_reward']:.1f}")
    print(f"  Improvement:   {((trained_rewards['avg_reward'] - random_rewards['avg_reward']) / random_rewards['avg_reward'] * 100):.1f}%")


def test_drone_performance(model_path, num_episodes=5):
    """Test drone performance without visualization."""
    model = PPO.load(model_path)
    env = RLWildfireEnv(grid_size=25, max_steps=500)
    
    rewards = []
    fires_extinguished = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        fires_ext = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            fires_ext = info['fires_extinguished']
            
            if terminated or truncated:
                break
        
        rewards.append(total_reward)
        fires_extinguished.append(fires_ext)
        print(f"  Episode {episode + 1}: Reward {total_reward:.1f}, Fires {fires_ext}")
    
    env.close()
    
    return {
        'avg_reward': sum(rewards) / len(rewards),
        'std_reward': (sum([(r - sum(rewards)/len(rewards))**2 for r in rewards]) / len(rewards))**0.5,
        'avg_fires': sum(fires_extinguished) / len(fires_extinguished)
    }


def test_random_drone(num_episodes=5):
    """Test random drone performance."""
    env = RLWildfireEnv(grid_size=25, max_steps=500)
    
    rewards = []
    fires_extinguished = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        fires_ext = 0
        
        while True:
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            fires_ext = info['fires_extinguished']
            
            if terminated or truncated:
                break
        
        rewards.append(total_reward)
        fires_extinguished.append(fires_ext)
        print(f"  Episode {episode + 1}: Reward {total_reward:.1f}, Fires {fires_ext}")
    
    env.close()
    
    return {
        'avg_reward': sum(rewards) / len(rewards),
        'std_reward': (sum([(r - sum(rewards)/len(rewards))**2 for r in rewards]) / len(rewards))**0.5,
        'avg_fires': sum(fires_extinguished) / len(fires_extinguished)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo PyroGuard AI Trained Drone')
    parser.add_argument('--model', type=str, default='models/final_model_ppo.zip',
                       help='Path to trained model')
    parser.add_argument('--compare', action='store_true',
                       help='Compare trained vs random performance')
    parser.add_argument('--no-render', action='store_true',
                       help='Run without visualization')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_with_random()
    else:
        if args.no_render:
            test_drone_performance(args.model, num_episodes=3)
        else:
            demo_trained_drone(args.model)
