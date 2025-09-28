#!/usr/bin/env python3
"""
Training script for PyroGuard AI RL Drone
Uses Stable Baselines3 to train a firefighting drone with focus on fire spread control.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from rl_wildfire_env import RLWildfireEnv
import gymnasium as gym


def create_training_environment(grid_size=25, max_steps=500):
    """Create the training environment."""
    def _init():
        env = RLWildfireEnv(grid_size=grid_size, max_steps=max_steps)
        env = Monitor(env)
        return env
    return _init


def train_drone(algorithm='PPO', total_timesteps=100000, save_path='./models/'):
    """Train the firefighting drone using specified algorithm."""
    
    print(f"🚁 Training PyroGuard AI Drone with {algorithm}")
    print("=" * 50)
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f"{save_path}/logs", exist_ok=True)
    
    # Create training environment
    print("🌲 Creating training environment...")
    train_env = make_vec_env(
        create_training_environment(grid_size=25, max_steps=500),
        n_envs=4,  # Parallel environments for faster training
        vec_env_cls=DummyVecEnv
    )
    
    # Create evaluation environment
    eval_env = make_vec_env(
        create_training_environment(grid_size=25, max_steps=500),
        n_envs=1,
        vec_env_cls=DummyVecEnv
    )
    
    # Initialize the model
    if algorithm == 'PPO':
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            learning_rate=1e-4,        # Reduced for more stable learning
            n_steps=2048,
            batch_size=32,             # Smaller batch for better gradient estimates
            n_epochs=15,               # More epochs for better learning
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.15,           # Tighter clipping for stability
            ent_coef=0.02,             # Higher entropy for more exploration
            vf_coef=0.3,               # Lower value function coefficient
            policy_kwargs=dict(
                net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # Larger networks
            ),
            tensorboard_log=f"{save_path}/logs"
        )
    elif algorithm == 'A2C':
        model = A2C(
            "MlpPolicy",
            train_env,
            verbose=1,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.01,
            vf_coef=0.25,
            tensorboard_log=f"{save_path}/logs"
        )
    elif algorithm == 'DQN':
        model = DQN(
            "MlpPolicy",
            train_env,
            verbose=1,
            learning_rate=1e-4,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            tensorboard_log=f"{save_path}/logs"
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Set up evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}/best_model",
        log_path=f"{save_path}/logs",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Train the model
    print(f"🎯 Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save the final model
    model_path = f"{save_path}/final_model_{algorithm.lower()}"
    model.save(model_path)
    print(f"💾 Model saved to {model_path}")
    
    # Clean up
    train_env.close()
    eval_env.close()
    
    return model


def evaluate_model(model_path, num_episodes=10, render=False):
    """Evaluate a trained model."""
    print(f"📊 Evaluating model: {model_path}")
    print("=" * 30)
    
    # Load the model
    if 'ppo' in model_path.lower():
        model = PPO.load(model_path)
    elif 'a2c' in model_path.lower():
        model = A2C.load(model_path)
    elif 'dqn' in model_path.lower():
        model = DQN.load(model_path)
    else:
        # Try to auto-detect
        try:
            model = PPO.load(model_path)
        except:
            try:
                model = A2C.load(model_path)
            except:
                model = DQN.load(model_path)
    
    # Create evaluation environment
    env = RLWildfireEnv(grid_size=25, max_steps=500)
    
    episode_rewards = []
    episode_fires_extinguished = []
    episode_steps = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        fires_extinguished = 0
        step = 0
        
        print(f"Episode {episode + 1}/{num_episodes}")
        
        while True:
            if render:
                env.render()
            
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            fires_extinguished = info.get('fires_extinguished', 0)
            step += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        episode_fires_extinguished.append(fires_extinguished)
        episode_steps.append(step)
        
        print(f"  Reward: {total_reward:.2f}, Fires Extinguished: {fires_extinguished}, Steps: {step}")
    
    env.close()
    
    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    avg_fires = np.mean(episode_fires_extinguished)
    avg_steps = np.mean(episode_steps)
    
    print(f"\n📈 Evaluation Results:")
    print(f"  Average Reward: {avg_reward:.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Average Fires Extinguished: {avg_fires:.2f} ± {np.std(episode_fires_extinguished):.2f}")
    print(f"  Average Steps: {avg_steps:.2f} ± {np.std(episode_steps):.2f}")
    
    return {
        'rewards': episode_rewards,
        'fires_extinguished': episode_fires_extinguished,
        'steps': episode_steps
    }


def compare_algorithms():
    """Compare different RL algorithms."""
    algorithms = ['PPO', 'A2C', 'DQN']
    results = {}
    
    for algo in algorithms:
        print(f"\n🔄 Training {algo}...")
        model = train_drone(algorithm=algo, total_timesteps=50000)
        
        # Evaluate
        model_path = f"./models/final_model_{algo.lower()}"
        results[algo] = evaluate_model(model_path, num_episodes=5)
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.boxplot([results[algo]['rewards'] for algo in algorithms], labels=algorithms)
    plt.title('Episode Rewards')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 3, 2)
    plt.boxplot([results[algo]['fires_extinguished'] for algo in algorithms], labels=algorithms)
    plt.title('Fires Extinguished')
    plt.ylabel('Number of Fires')
    
    plt.subplot(1, 3, 3)
    plt.boxplot([results[algo]['steps'] for algo in algorithms], labels=algorithms)
    plt.title('Episode Length')
    plt.ylabel('Steps')
    
    plt.tight_layout()
    plt.savefig('./models/algorithm_comparison.png')
    plt.show()
    
    return results


def demo_trained_drone(model_path, render=True):
    """Demo a trained drone."""
    print(f"🎮 Demo: Trained Drone from {model_path}")
    
    # Load model
    if 'ppo' in model_path.lower():
        model = PPO.load(model_path)
    elif 'a2c' in model_path.lower():
        model = A2C.load(model_path)
    elif 'dqn' in model_path.lower():
        model = DQN.load(model_path)
    else:
        model = PPO.load(model_path)  # Default
    
    # Create environment
    env = RLWildfireEnv(grid_size=30, max_steps=1000)
    
    obs, _ = env.reset()
    total_reward = 0
    step = 0
    
    print("🎯 Starting demo... (Close window to stop)")
    
    try:
        while True:
            if render:
                env.render()
            
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step += 1
            
            # Print periodic updates
            if step % 50 == 0:
                print(f"Step {step}: Reward {total_reward:.2f}, "
                      f"Fires: {info['fires_active']}, "
                      f"Extinguished: {info['fires_extinguished']}")
            
            if terminated or truncated:
                print(f"🎉 Episode complete! Total reward: {total_reward:.2f}")
                break
                
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    finally:
        env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PyroGuard AI Drone')
    parser.add_argument('--algorithm', type=str, default='PPO', 
                       choices=['PPO', 'A2C', 'DQN'], help='RL Algorithm')
    parser.add_argument('--timesteps', type=int, default=100000, 
                       help='Training timesteps')
    parser.add_argument('--eval', type=str, default=None, 
                       help='Evaluate trained model')
    parser.add_argument('--demo', type=str, default=None, 
                       help='Demo trained model')
    parser.add_argument('--compare', action='store_true', 
                       help='Compare all algorithms')
    
    args = parser.parse_args()
    
    if args.compare:
        print("🔄 Comparing all algorithms...")
        compare_algorithms()
    elif args.eval:
        print(f"📊 Evaluating {args.eval}...")
        evaluate_model(args.eval, num_episodes=10)
    elif args.demo:
        print(f"🎮 Demo {args.demo}...")
        demo_trained_drone(args.demo, render=True)
    else:
        print(f"🚁 Training {args.algorithm} for {args.timesteps} timesteps...")
        train_drone(algorithm=args.algorithm, total_timesteps=args.timesteps)
