#!/usr/bin/env python3
"""
RL Drone Training Script for PyroGuard AI

This script trains a Deep Q-Network (DQN) agent to control firefighter drones
in wildfire suppression scenarios.

Usage:
    python train_rl_drone.py [--episodes 5000] [--save-freq 500] [--eval-freq 100]
"""

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List
import json

# Add parent directory to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our RL components
from drones.rl.rl_environment import RLWildfireEnvironment
from drones.rl.rl_agent import RLFirefighterDrone, create_rl_drone, evaluate_drone


class TrainingLogger:
    """Logger for training metrics and visualization."""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.losses = []
        self.epsilon_values = []
        self.evaluation_results = []
        
        # Timing
        self.start_time = time.time()
        self.episode_times = []
    
    def log_episode(self, episode: int, reward: float, length: int, 
                   success: bool, drone_stats: Dict, episode_time: float):
        """Log data from completed episode."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.success_rates.append(1.0 if success else 0.0)
        self.epsilon_values.append(drone_stats['epsilon'])
        self.episode_times.append(episode_time)
        
        if drone_stats['recent_loss'] > 0:
            self.losses.append(drone_stats['recent_loss'])
        
        # Print progress every 100 episodes
        if episode % 100 == 0:
            self._print_progress(episode, drone_stats)
    
    def log_evaluation(self, episode: int, eval_results: Dict):
        """Log evaluation results."""
        eval_data = {'episode': episode, **eval_results}
        self.evaluation_results.append(eval_data)
        
        print(f"📊 Evaluation at episode {episode}:")
        print(f"   Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"   Success Rate: {eval_results['success_rate']:.2%}")
        print(f"   Mean Episode Length: {eval_results['mean_length']:.1f}")
    
    def _print_progress(self, episode: int, drone_stats: Dict):
        """Print training progress."""
        recent_rewards = self.episode_rewards[-100:]
        recent_success = self.success_rates[-100:]
        recent_lengths = self.episode_lengths[-100:]
        
        elapsed_time = time.time() - self.start_time
        episodes_per_sec = episode / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n🔥 Episode {episode} Progress Report:")
        print(f"   Recent Reward (100ep): {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}")
        print(f"   Recent Success Rate: {np.mean(recent_success):.2%}")
        print(f"   Recent Episode Length: {np.mean(recent_lengths):.1f}")
        print(f"   Current Epsilon: {drone_stats['epsilon']:.4f}")
        print(f"   Buffer Size: {drone_stats['buffer_size']}")
        print(f"   Recent Loss: {drone_stats['recent_loss']:.4f}")
        print(f"   Training Speed: {episodes_per_sec:.2f} episodes/sec")
        print(f"   Total Training Time: {elapsed_time/60:.1f} minutes")
    
    def save_plots(self):
        """Save training plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RL Firefighter Drone Training Results', fontsize=16)
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, color='blue')
        if len(self.episode_rewards) > 50:
            # Moving average
            window = min(100, len(self.episode_rewards) // 10)
            moving_avg = np.convolve(self.episode_rewards, 
                                   np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(self.episode_rewards)), 
                          moving_avg, color='red', linewidth=2)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Success rate
        if len(self.success_rates) > 50:
            window = min(100, len(self.success_rates) // 10)
            success_avg = np.convolve(self.success_rates, 
                                    np.ones(window)/window, mode='valid')
            axes[0, 1].plot(range(window-1, len(self.success_rates)), 
                          success_avg, color='green', linewidth=2)
        axes[0, 1].set_title('Success Rate (Moving Average)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True)
        
        # Episode lengths
        axes[0, 2].plot(self.episode_lengths, alpha=0.3, color='purple')
        if len(self.episode_lengths) > 50:
            window = min(100, len(self.episode_lengths) // 10)
            length_avg = np.convolve(self.episode_lengths, 
                                   np.ones(window)/window, mode='valid')
            axes[0, 2].plot(range(window-1, len(self.episode_lengths)), 
                          length_avg, color='orange', linewidth=2)
        axes[0, 2].set_title('Episode Lengths')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Steps')
        axes[0, 2].grid(True)
        
        # Epsilon decay
        axes[1, 0].plot(self.epsilon_values, color='red')
        axes[1, 0].set_title('Exploration Rate (Epsilon)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].grid(True)
        
        # Training loss
        if self.losses:
            axes[1, 1].plot(self.losses, alpha=0.7, color='brown')
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        
        # Evaluation results
        if self.evaluation_results:
            eval_episodes = [r['episode'] for r in self.evaluation_results]
            eval_rewards = [r['mean_reward'] for r in self.evaluation_results]
            eval_success = [r['success_rate'] for r in self.evaluation_results]
            
            ax_eval = axes[1, 2]
            color1 = 'tab:blue'
            ax_eval.set_xlabel('Episode')
            ax_eval.set_ylabel('Evaluation Reward', color=color1)
            ax_eval.plot(eval_episodes, eval_rewards, color=color1, marker='o')
            ax_eval.tick_params(axis='y', labelcolor=color1)
            ax_eval.grid(True)
            
            ax_eval2 = ax_eval.twinx()
            color2 = 'tab:green'
            ax_eval2.set_ylabel('Success Rate', color=color2)
            ax_eval2.plot(eval_episodes, eval_success, color=color2, marker='s')
            ax_eval2.tick_params(axis='y', labelcolor=color2)
            ax_eval2.set_ylim(0, 1)
            
            axes[1, 2].set_title('Evaluation Results')
        
        plt.tight_layout()
        plot_path = os.path.join(self.log_dir, 'training_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training plots saved to {plot_path}")
    
    def save_metrics(self):
        """Save training metrics to JSON."""
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_rates': self.success_rates,
            'epsilon_values': self.epsilon_values,
            'evaluation_results': self.evaluation_results,
            'total_episodes': len(self.episode_rewards),
            'total_training_time_minutes': (time.time() - self.start_time) / 60,
            'final_success_rate': np.mean(self.success_rates[-100:]) if len(self.success_rates) >= 100 else np.mean(self.success_rates),
            'final_average_reward': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
        }
        
        metrics_path = os.path.join(self.log_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"💾 Training metrics saved to {metrics_path}")


def train_rl_drone(num_episodes: int = 5000, 
                  save_freq: int = 500,
                  eval_freq: int = 100,
                  model_dir: str = "models",
                  log_dir: str = "logs"):
    """
    Main training function for RL firefighter drone.
    
    Args:
        num_episodes: Number of training episodes
        save_freq: Frequency to save model checkpoints
        eval_freq: Frequency to evaluate model performance
        model_dir: Directory to save model checkpoints
        log_dir: Directory to save logs and plots
    """
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"rl_drone_run_{timestamp}"
    model_path = os.path.join(model_dir, run_dir)
    log_path = os.path.join(log_dir, run_dir)
    
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    print(f"🚀 Starting RL Drone Training")
    print(f"   Episodes: {num_episodes}")
    print(f"   Model Path: {model_path}")
    print(f"   Log Path: {log_path}")
    print(f"   Save Frequency: {save_freq}")
    print(f"   Eval Frequency: {eval_freq}")
    
    # Initialize environment
    print("\n🌲 Initializing RL Environment...")
    env = RLWildfireEnvironment(
        grid_size=20,
        max_episode_steps=400,
        fire_spawn_rate=0.02
    )
    
    print(f"   Grid Size: {env.grid_size}x{env.grid_size}")
    print(f"   Max Episode Steps: {env.max_episode_steps}")
    print(f"   Observation Space: {env.observation_space}")
    print(f"   Action Space: {env.action_space}")
    
    # Initialize RL drone
    print("\n🤖 Initializing RL Drone...")
    drone_config = {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': int(num_episodes * 0.8),  # Decay over 80% of training
        'buffer_size': 50000,
        'batch_size': 64,
        'target_update_freq': 500
    }
    
    drone = create_rl_drone(env.observation_space, env.action_space, drone_config)
    
    # Initialize logger
    logger = TrainingLogger(log_path)
    
    print("\n🎯 Starting Training Loop...")
    
    try:
        for episode in range(num_episodes):
            episode_start_time = time.time()
            
            # Reset environment
            observation, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # Select action
                action = drone.select_action(observation, training=True)
                
                # Take step
                next_observation, reward, terminated, truncated, step_info = env.step(action)
                
                # Train drone
                training_stats = drone.train_step(observation, action, reward, 
                                                next_observation, terminated or truncated)
                
                # Update for next iteration
                observation = next_observation
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            # End episode
            success = step_info.get('is_successful', False)
            drone.end_episode(episode_reward, episode_length, success)
            
            # Log episode
            episode_time = time.time() - episode_start_time
            drone_stats = drone.get_statistics()
            logger.log_episode(episode, episode_reward, episode_length, 
                             success, drone_stats, episode_time)
            
            # Evaluation
            if (episode + 1) % eval_freq == 0:
                print(f"\n🧪 Evaluating model at episode {episode + 1}...")
                eval_results = evaluate_drone(drone, env, num_episodes=10)
                logger.log_evaluation(episode + 1, eval_results)
            
            # Save model
            if (episode + 1) % save_freq == 0:
                model_file = os.path.join(model_path, f"drone_model_ep_{episode + 1}.pth")
                drone.save_model(model_file)
        
        print("\n🎉 Training Completed Successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise
    
    finally:
        # Final evaluation
        print("\n🏁 Final Evaluation...")
        final_eval = evaluate_drone(drone, env, num_episodes=20)
        logger.log_evaluation(num_episodes, final_eval)
        
        # Save final model
        final_model_path = os.path.join(model_path, "final_drone_model.pth")
        drone.save_model(final_model_path)
        
        # Save training results
        logger.save_plots()
        logger.save_metrics()
        
        # Print final statistics
        print(f"\n📊 Final Training Statistics:")
        print(f"   Total Episodes: {len(logger.episode_rewards)}")
        print(f"   Final Success Rate: {final_eval['success_rate']:.2%}")
        print(f"   Final Mean Reward: {final_eval['mean_reward']:.2f}")
        print(f"   Total Training Time: {(time.time() - logger.start_time)/60:.1f} minutes")
        
        # Close environment
        env.close()
        
        print(f"\n✅ All results saved to:")
        print(f"   Models: {model_path}")
        print(f"   Logs: {log_path}")


def continue_training(model_path: str, num_additional_episodes: int = 1000):
    """Continue training from a saved model."""
    print(f"🔄 Continuing training from {model_path}")
    
    # Initialize environment and drone
    env = RLWildfireEnvironment(grid_size=20, max_episode_steps=400)
    drone = create_rl_drone(env.observation_space, env.action_space)
    
    # Load model
    drone.load_model(model_path)
    
    # Continue training
    train_rl_drone(
        num_episodes=num_additional_episodes,
        save_freq=100,
        eval_freq=50,
        model_dir="models_continued",
        log_dir="logs_continued"
    )


def test_trained_model(model_path: str, num_test_episodes: int = 20, render: bool = True):
    """Test a trained model."""
    print(f"🧪 Testing trained model: {model_path}")
    
    # Initialize environment and drone
    env = RLWildfireEnvironment(grid_size=20, max_episode_steps=400)
    drone = create_rl_drone(env.observation_space, env.action_space)
    
    # Load trained model
    drone.load_model(model_path)
    
    # Test episodes
    total_reward = 0
    successful_episodes = 0
    
    for episode in range(num_test_episodes):
        observation, _ = env.reset()
        episode_reward = 0
        
        print(f"\n🎮 Test Episode {episode + 1}")
        
        while True:
            # Select action (no exploration)
            action = drone.select_action(observation, training=False)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if render and episode < 3:  # Render first few episodes
                env.render()
                time.sleep(0.1)  # Slow down for visualization
            
            if terminated or truncated:
                success = info.get('is_successful', False)
                if success:
                    successful_episodes += 1
                    print(f"   ✅ Success! Reward: {episode_reward:.2f}")
                else:
                    print(f"   ❌ Failed. Reward: {episode_reward:.2f}")
                break
        
        total_reward += episode_reward
    
    # Final results
    avg_reward = total_reward / num_test_episodes
    success_rate = successful_episodes / num_test_episodes
    
    print(f"\n📊 Test Results:")
    print(f"   Episodes: {num_test_episodes}")
    print(f"   Average Reward: {avg_reward:.2f}")
    print(f"   Success Rate: {success_rate:.2%}")
    
    env.close()


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Train RL Firefighter Drone')
    parser.add_argument('--episodes', type=int, default=3000,
                       help='Number of training episodes (default: 3000)')
    parser.add_argument('--save-freq', type=int, default=300,
                       help='Model save frequency (default: 300)')
    parser.add_argument('--eval-freq', type=int, default=100,
                       help='Evaluation frequency (default: 100)')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Model save directory (default: models)')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Log save directory (default: logs)')
    parser.add_argument('--continue-from', type=str, default=None,
                       help='Path to model to continue training from')
    parser.add_argument('--test-model', type=str, default=None,
                       help='Path to model to test')
    parser.add_argument('--test-episodes', type=int, default=10,
                       help='Number of test episodes (default: 10)')
    
    args = parser.parse_args()
    
    if args.test_model:
        # Test mode
        test_trained_model(args.test_model, args.test_episodes)
    elif args.continue_from:
        # Continue training mode
        continue_training(args.continue_from, args.episodes)
    else:
        # Normal training mode
        train_rl_drone(
            num_episodes=args.episodes,
            save_freq=args.save_freq,
            eval_freq=args.eval_freq,
            model_dir=args.model_dir,
            log_dir=args.log_dir
        )


if __name__ == "__main__":
    main()