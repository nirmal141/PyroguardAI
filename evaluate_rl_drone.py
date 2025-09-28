#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for PyroGuard AI RL Drone
Evaluates training sufficiency and compares different approaches.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO, A2C, DQN
from rl_wildfire_env import RLWildfireEnv
import os
import time
from typing import Dict, List, Tuple


class RLEvaluator:
    """Comprehensive RL evaluation framework."""
    
    def __init__(self, grid_size=30, num_episodes=20):
        self.grid_size = grid_size
        self.num_episodes = num_episodes
        self.results = {}
    
    def evaluate_model(self, model_path: str, model_name: str = "Trained") -> Dict:
        """Evaluate a trained model comprehensively."""
        print(f"📊 Evaluating {model_name}...")
        
        # Load model
        try:
            if 'ppo' in model_path.lower():
                model = PPO.load(model_path)
            elif 'a2c' in model_path.lower():
                model = A2C.load(model_path)
            elif 'dqn' in model_path.lower():
                model = DQN.load(model_path)
            else:
                model = PPO.load(model_path)  # Default
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return {}
        
        # Run evaluation episodes
        episodes_data = []
        
        for episode in range(self.num_episodes):
            env = RLWildfireEnv(grid_size=self.grid_size, max_steps=1000)
            obs, _ = env.reset()
            
            episode_data = {
                'episode': episode,
                'total_reward': 0,
                'fires_extinguished': 0,
                'steps_taken': 0,
                'energy_used': 0,
                'water_used': 0,
                'base_time': 0,
                'exploration_efficiency': 0,
                'success': False
            }
            
            step = 0
            initial_energy = 100
            initial_water = 20
            base_steps = 0
            
            while True:
                # Get action from model
                action, _ = model.predict(obs, deterministic=True)
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Track metrics
                episode_data['total_reward'] += reward
                episode_data['steps_taken'] += 1
                step += 1
                
                # Track base camping
                if info['drone_energy'] > 90 and info['drone_water'] > 18:  # At base
                    base_steps += 1
                
                # Check termination
                if terminated:
                    episode_data['success'] = True
                    episode_data['fires_extinguished'] = info['fires_extinguished']
                    break
                elif truncated:
                    episode_data['success'] = False
                    break
            
            # Calculate final metrics
            episode_data['energy_used'] = initial_energy - info['drone_energy']
            episode_data['water_used'] = initial_water - info['drone_water']
            episode_data['base_time'] = base_steps
            episode_data['exploration_efficiency'] = episode_data['fires_extinguished'] / max(episode_data['steps_taken'], 1)
            
            episodes_data.append(episode_data)
            env.close()
            
            if episode % 5 == 0:
                print(f"  Episode {episode + 1}/{self.num_episodes}: "
                      f"Reward {episode_data['total_reward']:.1f}, "
                      f"Fires {episode_data['fires_extinguished']}, "
                      f"Success {episode_data['success']}")
        
        # Calculate statistics
        rewards = [ep['total_reward'] for ep in episodes_data]
        fires_extinguished = [ep['fires_extinguished'] for ep in episodes_data]
        steps_taken = [ep['steps_taken'] for ep in episodes_data]
        success_rate = sum([ep['success'] for ep in episodes_data]) / len(episodes_data)
        base_time = [ep['base_time'] for ep in episodes_data]
        exploration_efficiency = [ep['exploration_efficiency'] for ep in episodes_data]
        
        results = {
            'model_name': model_name,
            'episodes': episodes_data,
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_fires_extinguished': np.mean(fires_extinguished),
            'std_fires_extinguished': np.std(fires_extinguished),
            'avg_steps': np.mean(steps_taken),
            'std_steps': np.std(steps_taken),
            'success_rate': success_rate,
            'avg_base_time': np.mean(base_time),
            'avg_exploration_efficiency': np.mean(exploration_efficiency),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards)
        }
        
        self.results[model_name] = results
        return results
    
    def evaluate_random_baseline(self) -> Dict:
        """Evaluate random baseline for comparison."""
        print("🎲 Evaluating Random Baseline...")
        
        episodes_data = []
        
        for episode in range(self.num_episodes):
            env = RLWildfireEnv(grid_size=self.grid_size, max_steps=1000)
            obs, _ = env.reset()
            
            episode_data = {
                'episode': episode,
                'total_reward': 0,
                'fires_extinguished': 0,
                'steps_taken': 0,
                'energy_used': 0,
                'water_used': 0,
                'base_time': 0,
                'exploration_efficiency': 0,
                'success': False
            }
            
            step = 0
            initial_energy = 100
            initial_water = 20
            base_steps = 0
            
            while True:
                # Random action
                action = env.action_space.sample()
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Track metrics
                episode_data['total_reward'] += reward
                episode_data['steps_taken'] += 1
                step += 1
                
                # Track base camping
                if info['drone_energy'] > 90 and info['drone_water'] > 18:
                    base_steps += 1
                
                # Check termination
                if terminated:
                    episode_data['success'] = True
                    episode_data['fires_extinguished'] = info['fires_extinguished']
                    break
                elif truncated:
                    episode_data['success'] = False
                    break
            
            # Calculate final metrics
            episode_data['energy_used'] = initial_energy - info['drone_energy']
            episode_data['water_used'] = initial_water - info['drone_water']
            episode_data['base_time'] = base_steps
            episode_data['exploration_efficiency'] = episode_data['fires_extinguished'] / max(episode_data['steps_taken'], 1)
            
            episodes_data.append(episode_data)
            env.close()
            
            if episode % 5 == 0:
                print(f"  Episode {episode + 1}/{self.num_episodes}: "
                      f"Reward {episode_data['total_reward']:.1f}, "
                      f"Fires {episode_data['fires_extinguished']}, "
                      f"Success {episode_data['success']}")
        
        # Calculate statistics
        rewards = [ep['total_reward'] for ep in episodes_data]
        fires_extinguished = [ep['fires_extinguished'] for ep in episodes_data]
        steps_taken = [ep['steps_taken'] for ep in episodes_data]
        success_rate = sum([ep['success'] for ep in episodes_data]) / len(episodes_data)
        base_time = [ep['base_time'] for ep in episodes_data]
        exploration_efficiency = [ep['exploration_efficiency'] for ep in episodes_data]
        
        results = {
            'model_name': 'Random',
            'episodes': episodes_data,
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_fires_extinguished': np.mean(fires_extinguished),
            'std_fires_extinguished': np.std(fires_extinguished),
            'avg_steps': np.mean(steps_taken),
            'std_steps': np.std(steps_taken),
            'success_rate': success_rate,
            'avg_base_time': np.mean(base_time),
            'avg_exploration_efficiency': np.mean(exploration_efficiency),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards)
        }
        
        self.results['Random'] = results
        return results
    
    def generate_report(self) -> str:
        """Generate comprehensive evaluation report."""
        if not self.results:
            return "No results to report."
        
        report = []
        report.append("🔥 PyroGuard AI - RL Training Evaluation Report")
        report.append("=" * 60)
        report.append("")
        
        # Summary table
        report.append("📊 Performance Summary:")
        report.append("-" * 40)
        report.append(f"{'Model':<15} {'Avg Reward':<12} {'Success Rate':<12} {'Fires/Episode':<12} {'Base Time':<10}")
        report.append("-" * 40)
        
        for model_name, results in self.results.items():
            report.append(f"{model_name:<15} {results['avg_reward']:<12.1f} {results['success_rate']:<12.1%} "
                         f"{results['avg_fires_extinguished']:<12.1f} {results['avg_base_time']:<10.1f}")
        
        report.append("")
        
        # Detailed analysis
        if 'Trained' in self.results and 'Random' in self.results:
            trained = self.results['Trained']
            random = self.results['Random']
            
            reward_improvement = ((trained['avg_reward'] - random['avg_reward']) / random['avg_reward']) * 100
            success_improvement = ((trained['success_rate'] - random['success_rate']) / random['success_rate']) * 100
            fires_improvement = ((trained['avg_fires_extinguished'] - random['avg_fires_extinguished']) / random['avg_fires_extinguished']) * 100
            
            report.append("🎯 Training Effectiveness Analysis:")
            report.append("-" * 40)
            report.append(f"Reward Improvement: {reward_improvement:+.1f}%")
            report.append(f"Success Rate Improvement: {success_improvement:+.1f}%")
            report.append(f"Fire Fighting Improvement: {fires_improvement:+.1f}%")
            report.append(f"Base Camping Reduction: {random['avg_base_time'] - trained['avg_base_time']:.1f} steps")
            report.append("")
            
            # Training sufficiency assessment
            report.append("✅ Training Sufficiency Assessment:")
            report.append("-" * 40)
            
            if reward_improvement > 20:
                report.append("✅ GOOD: Significant reward improvement over random")
            else:
                report.append("⚠️  WEAK: Limited reward improvement over random")
            
            if success_improvement > 10:
                report.append("✅ GOOD: Significant success rate improvement")
            else:
                report.append("⚠️  WEAK: Limited success rate improvement")
            
            if trained['avg_base_time'] < 5:
                report.append("✅ GOOD: Low base camping time")
            else:
                report.append("⚠️  WEAK: High base camping time")
            
            if trained['success_rate'] > 0.8:
                report.append("✅ GOOD: High success rate")
            else:
                report.append("⚠️  WEAK: Low success rate")
        
        return "\n".join(report)
    
    def plot_comparison(self, save_path="evaluation_plots.png"):
        """Generate comparison plots."""
        if len(self.results) < 2:
            print("Need at least 2 models to plot comparison")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('PyroGuard AI - RL Training Evaluation', fontsize=16)
        
        models = list(self.results.keys())
        metrics = ['avg_reward', 'success_rate', 'avg_fires_extinguished', 
                  'avg_steps', 'avg_base_time', 'avg_exploration_efficiency']
        metric_labels = ['Average Reward', 'Success Rate', 'Fires Extinguished',
                        'Average Steps', 'Base Time', 'Exploration Efficiency']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i//3, i%3]
            
            values = [self.results[model][metric] for model in models]
            stds = [self.results[model][f'std_{metric.split("_")[1]}' if '_' in metric else 'std_reward'] 
                   for model in models]
            
            bars = ax.bar(models, values, yerr=stds, capsize=5, alpha=0.7)
            ax.set_title(label)
            ax.set_ylabel(label)
            
            # Color bars differently
            for j, bar in enumerate(bars):
                if models[j] == 'Random':
                    bar.set_color('red')
                else:
                    bar.set_color('green')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Plots saved to {save_path}")


def main():
    """Run comprehensive evaluation."""
    print("🔥 PyroGuard AI - Comprehensive RL Evaluation")
    print("=" * 50)
    
    evaluator = RLEvaluator(grid_size=30, num_episodes=10)
    
    # Evaluate random baseline
    evaluator.evaluate_random_baseline()
    
    # Evaluate trained model
    if os.path.exists("models/final_model_ppo.zip"):
        evaluator.evaluate_model("models/final_model_ppo.zip", "Trained PPO")
    else:
        print("❌ No trained model found at models/final_model_ppo.zip")
        return
    
    # Generate report
    report = evaluator.generate_report()
    print(report)
    
    # Save report
    with open("evaluation_report.txt", "w") as f:
        f.write(report)
    print("📄 Report saved to evaluation_report.txt")
    
    # Generate plots
    evaluator.plot_comparison()


if __name__ == "__main__":
    main()
