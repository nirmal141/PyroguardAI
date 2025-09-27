"""
RL training script for multi-agent drone firefighting.
Uses stable-baselines3 for training PPO agents.
"""

import os
import sys
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import matplotlib.pyplot as plt

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.multi_agent_fire_env import MultiAgentFireEnv
from envs.drone_agent import DroneAction

class DroneFirefightingWrapper(gym.Env):
    """
    Wrapper to make the multi-agent environment compatible with stable-baselines3.
    This treats the multi-agent problem as a single-agent problem with vectorized actions.
    """
    
    def __init__(self, env):
        self.env = env
        # Initialize cell_size if not present
        if not hasattr(env, 'cell_size'):
            env.cell_size = 40
        # Flatten the multi-agent action space
        self.action_space = gym.spaces.MultiDiscrete([11] * env.num_drones)
        # Flatten the multi-agent observation space
        drone_obs_size = 6 + 9  # 6 state vars + 3x3 local grid
        total_obs_size = env.num_drones * drone_obs_size
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(total_obs_size,), dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # Flatten observations for single agent
        return obs.flatten(), info
    
    def step(self, action):
        # Convert flattened action back to multi-agent format
        if isinstance(action, np.ndarray):
            if action.ndim == 1 and len(action) == self.env.num_drones:
                # Already in correct format
                multi_agent_action = action
            else:
                # Reshape if needed
                multi_agent_action = action.reshape(self.env.num_drones)
        else:
            # Single action for all drones
            multi_agent_action = np.array([action] * self.env.num_drones)
        
        obs, reward, terminated, truncated, info = self.env.step(multi_agent_action)
        # Flatten observations
        obs = obs.flatten()
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode="human"):
        if mode == "human":
            # Initialize pygame if not already done
            if not hasattr(self.env, "screen") or self.env.screen is None:
                import pygame
                pygame.init()
                self.env.screen = pygame.display.set_mode((
                    self.env.width * self.env.cell_size, 
                    self.env.height * self.env.cell_size + 100
                ))
        return self.env.render(mode)
    
    def close(self):
        return self.env.close()

def make_env(env_id, rank=0, seed=0):
    """Create environment for vectorized training."""
    def _init():
        env = MultiAgentFireEnv(
            width=12, 
            height=12, 
            p_spread=0.35, 
            p_burnout=0.12, 
            max_steps=500, 
            ignitions=(1, 3),
            num_drones=2,  # Start with 2 drones for easier training
            seed=seed + rank
        )
        env = DroneFirefightingWrapper(env)
        return env
    return _init

def train_ppo_agent(total_timesteps=50000, n_envs=2):
    """Train a PPO agent for drone firefighting."""
    
    print("🚁 Starting RL training for drone firefighting...")
    
    # Create vectorized environment
    env = make_vec_env(
        make_env("DroneFirefighting-v0"), 
        n_envs=n_envs, 
        vec_env_cls=DummyVecEnv
    )
    
    # Create evaluation environment
    eval_env = make_vec_env(
        make_env("DroneFirefighting-v0"), 
        n_envs=1, 
        vec_env_cls=DummyVecEnv
    )
    
    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5
    )
    
    # Set up callbacks
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path="./models/drone_firefighting/",
        log_path="./logs/drone_firefighting/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Train the agent
    print(f"🎯 Training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps
    )
    
    # Save the final model
    model.save("./models/drone_firefighting/final_model")
    print("✅ Training completed! Model saved.")
    
    return model

def test_trained_agent(model_path="./models/drone_firefighting/best_model.zip"):
    """Test the trained agent."""
    
    print("🧪 Testing trained agent...")
    
    # Create test environment (match training config)
    env = MultiAgentFireEnv(
        width=12, 
        height=12, 
        p_spread=0.35, 
        p_burnout=0.12, 
        max_steps=500, 
        ignitions=(1, 3),
        num_drones=2,  # Match training config
        seed=123
    )
    
    env = DroneFirefightingWrapper(env)
    
    # Load trained model
    try:
        model = PPO.load(model_path)
        print(f"✅ Loaded model from {model_path}")
    except:
        print(f"❌ Could not load model from {model_path}")
        print("Training a new model instead...")
        return train_ppo_agent(total_timesteps=50000, n_envs=2)
    
    # Test the agent
    obs, _ = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    
    # Activate firefighting mode
    env.env.activate_drone_firefighting()
    
    print("🎮 Starting test run...")
    print("Controls: Press 'T' to toggle firefighting, 'R' to recharge drones")
    
    import pygame
    pygame.init()
    clock = pygame.time.Clock()
    
    try:
        while not done and step_count < 1000:
            # Get action from trained model
            # Reshape observation to match training format
            obs_reshaped = obs.reshape(1, -1) if obs.ndim == 1 else obs
            action, _ = model.predict(obs_reshaped, deterministic=True)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Render
            env.render()
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_t:
                        if env.env.drone_firefighting_active:
                            env.env.deactivate_drone_firefighting()
                        else:
                            env.env.activate_drone_firefighting()
                    elif event.key == pygame.K_r:
                        env.env.recharge_drones()
            
            # Print status every 50 steps
            if step_count % 50 == 0:
                print(f"Step {step_count}: Reward={reward:.2f}, "
                      f"Total={total_reward:.2f}, Fires={info['burning']}")
            
            clock.tick(10)  # 10 FPS
            
            done = terminated or truncated
    
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    
    finally:
        pygame.quit()
    
    print(f"\n🎯 Test Results:")
    print(f"  Steps: {step_count}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Fires Extinguished: {env.env.total_fires_extinguished}")
    
    return model

def main():
    """Main training and testing function."""
    
    # Create directories
    os.makedirs("./models/drone_firefighting/", exist_ok=True)
    os.makedirs("./logs/drone_firefighting/", exist_ok=True)
    
    print("🔥 Multi-Agent Drone Firefighting RL Training")
    print("=" * 60)
    
    # Check if we want to train or test
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode
        model_path = sys.argv[2] if len(sys.argv) > 2 else "./models/drone_firefighting/best_model.zip"
        test_trained_agent(model_path)
    else:
        # Training mode
        print("🚀 Starting training...")
        model = train_ppo_agent(total_timesteps=50000, n_envs=2)
        
        print("\n🧪 Testing trained model...")
        test_trained_agent("./models/drone_firefighting/best_model.zip")

if __name__ == "__main__":
    main()
