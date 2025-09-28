import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional
import math


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for firefighter drone control.
    
    Uses CNN for processing local terrain/fire views and fully connected layers
    for integrating with drone state and global information.
    """
    
    def __init__(self, observation_space, action_space, hidden_size: int = 512):
        super(DQNNetwork, self).__init__()
        
        # Get dimensions from observation space
        local_terrain_shape = observation_space['local_terrain'].shape  # (11, 11)
        drone_state_dim = observation_space['drone_state'].shape[0]     # 6
        global_state_dim = observation_space['global_state'].shape[0]   # 5
        
        # CNN for processing local terrain view
        self.terrain_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Reduce to 4x4
        )
        
        # CNN for processing local fire age view
        self.fire_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Reduce to 4x4
        )
        
        # Calculate CNN output dimensions
        cnn_output_dim = 64 * 4 * 4 + 32 * 4 * 4  # terrain + fire CNNs
        
        # Fully connected layers for state integration
        self.state_integration = nn.Sequential(
            nn.Linear(cnn_output_dim + drone_state_dim + global_state_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        # Separate advantage and value streams (Dueling DQN)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, action_space.nvec[0] * action_space.nvec[1])
        )
        
        # Action space dimensions
        self.num_move_actions = action_space.nvec[0]  # 9 movement directions
        self.num_action_types = action_space.nvec[1]  # 3 action types
        self.total_actions = self.num_move_actions * self.num_action_types
        
    def forward(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the network."""
        batch_size = observation['local_terrain'].shape[0]
        
        # Process local terrain view
        terrain_input = observation['local_terrain'].unsqueeze(1).float()  # Add channel dim
        terrain_features = self.terrain_cnn(terrain_input)
        terrain_features = terrain_features.view(batch_size, -1)
        
        # Process local fire age view
        fire_input = observation['local_fire_age'].unsqueeze(1).float()  # Add channel dim
        fire_features = self.fire_cnn(fire_input)
        fire_features = fire_features.view(batch_size, -1)
        
        # Concatenate all features
        combined_features = torch.cat([
            terrain_features,
            fire_features,
            observation['drone_state'],
            observation['global_state']
        ], dim=1)
        
        # Process through state integration layers
        integrated_state = self.state_integration(combined_features)
        
        # Dueling DQN: separate value and advantage
        values = self.value_stream(integrated_state)
        advantages = self.advantage_stream(integrated_state)
        
        # Combine value and advantage
        advantages = advantages.view(batch_size, self.total_actions)
        advantages_mean = advantages.mean(dim=1, keepdim=True)
        q_values = values + (advantages - advantages_mean)
        
        return q_values
    
    def get_action_probs(self, observation: Dict[str, torch.Tensor], temperature: float = 1.0) -> torch.Tensor:
        """Get action probabilities using softmax with temperature."""
        q_values = self.forward(observation)
        return F.softmax(q_values / temperature, dim=1)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer for more efficient learning.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
        
        # Experience tuple
        self.Experience = namedtuple('Experience', 
                                   ['state', 'action', 'reward', 'next_state', 'done'])
    
    def add(self, state: Dict[str, np.ndarray], action: List[int], reward: float, 
            next_state: Dict[str, np.ndarray], done: bool):
        """Add experience to buffer."""
        experience = self.Experience(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # Assign maximum priority to new experience
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List, np.ndarray, np.ndarray]:
        """Sample batch with prioritized sampling."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, error in zip(indices, errors):
            priority = (abs(error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


class RLFirefighterDrone:
    """
    Reinforcement Learning Firefighter Drone using Deep Q-Network with improvements:
    - Dueling DQN architecture
    - Prioritized Experience Replay
    - Double DQN for stable learning
    - Epsilon-greedy exploration with decay
    - Reward shaping for faster learning
    """
    
    def __init__(self, observation_space, action_space, 
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: int = 10000,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 target_update_freq: int = 1000,
                 device: Optional[str] = None):
        """
        Initialize the RL Firefighter Drone.
        
        Args:
            observation_space: Gymnasium observation space
            action_space: Gymnasium action space  
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Steps over which to decay epsilon
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            device: Device to run model on ('cpu' or 'cuda')
        """
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🤖 RL Drone initialized on device: {self.device}")
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Exploration parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        
        # Networks
        self.q_network = DQNNetwork(observation_space, action_space).to(self.device)
        self.target_network = DQNNetwork(observation_space, action_space).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Copy weights to target network
        self.update_target_network()
        
        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        
        # Training statistics
        self.steps_done = 0
        self.episodes_trained = 0
        self.total_reward = 0.0
        self.losses = []
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_rate = deque(maxlen=100)
        
    def select_action(self, observation: Dict[str, np.ndarray], training: bool = True) -> List[int]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            observation: Current observation from environment
            training: Whether in training mode (affects exploration)
            
        Returns:
            List of action indices [move_direction, action_type]
        """
        
        # Convert observation to tensors
        obs_tensors = self._prepare_observation(observation)
        
        if training and random.random() < self.epsilon:
            # Random action (exploration)
            move_direction = random.randint(0, self.action_space.nvec[0] - 1)
            action_type = random.randint(0, self.action_space.nvec[1] - 1)
            return [move_direction, action_type]
        
        # Greedy action (exploitation)
        with torch.no_grad():
            q_values = self.q_network(obs_tensors)
            action_idx = q_values.argmax(dim=1).item()
            
            # Convert flat action index to [move_direction, action_type]
            move_direction = action_idx // self.action_space.nvec[1]
            action_type = action_idx % self.action_space.nvec[1]
            
            return [move_direction, action_type]
    
    def train_step(self, observation: Dict[str, np.ndarray], 
                   action: List[int], reward: float, 
                   next_observation: Dict[str, np.ndarray], 
                   done: bool) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            observation: Previous observation
            action: Action taken
            reward: Reward received
            next_observation: New observation
            done: Whether episode terminated
            
        Returns:
            Dictionary with training statistics
        """
        
        # Store experience in replay buffer
        self.replay_buffer.add(observation, action, reward, next_observation, done)
        
        self.steps_done += 1
        self.total_reward += reward
        
        # Update epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * 
            (self.steps_done / self.epsilon_decay)
        )
        
        training_stats = {'loss': 0.0, 'q_value_mean': 0.0}
        
        # Train if enough experiences in buffer
        if len(self.replay_buffer) >= self.batch_size:
            loss, q_value_mean = self._update_q_network()
            training_stats['loss'] = loss
            training_stats['q_value_mean'] = q_value_mean
        
        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.update_target_network()
            print(f"🎯 Target network updated at step {self.steps_done}")
        
        return training_stats
    
    def _update_q_network(self) -> Tuple[float, float]:
        """Update Q-network using batch of experiences."""
        # Sample batch from replay buffer
        experiences, indices, weights = self.replay_buffer.sample(self.batch_size)
        
        # Separate batch components
        states = [e.state for e in experiences]
        actions = [e.action for e in experiences]
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32, device=self.device)
        next_states = [e.next_state for e in experiences]
        dones = torch.tensor([e.done for e in experiences], dtype=torch.bool, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        
        # Prepare batch observations
        state_batch = self._prepare_batch_observations(states)
        next_state_batch = self._prepare_batch_observations(next_states)
        
        # Convert actions to tensor
        action_indices = []
        for action in actions:
            move_dir, action_type = action
            flat_action = move_dir * self.action_space.nvec[1] + action_type
            action_indices.append(flat_action)
        action_indices = torch.tensor(action_indices, dtype=torch.long, device=self.device)
        
        # Current Q values
        current_q_values = self.q_network(state_batch).gather(1, action_indices.unsqueeze(1))
        
        # Next Q values using Double DQN
        with torch.no_grad():
            # Use main network to select actions
            next_actions = self.q_network(next_state_batch).argmax(dim=1)
            # Use target network to evaluate actions
            next_q_values = self.target_network(next_state_batch).gather(1, next_actions.unsqueeze(1))
            next_q_values = next_q_values.squeeze(1)
            
            # Calculate target Q values
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Calculate loss with importance sampling weights
        td_errors = current_q_values.squeeze(1) - target_q_values
        loss = (weights * (td_errors ** 2)).mean()
        
        # Update priorities in replay buffer
        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Statistics
        loss_value = loss.item()
        q_value_mean = current_q_values.mean().item()
        
        self.losses.append(loss_value)
        
        return loss_value, q_value_mean
    
    def _prepare_observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert single observation to tensor format."""
        obs_tensors = {}
        for key, value in observation.items():
            tensor = torch.tensor(value, dtype=torch.float32, device=self.device)
            # Add batch dimension
            obs_tensors[key] = tensor.unsqueeze(0)
        return obs_tensors
    
    def _prepare_batch_observations(self, observations: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """Convert batch of observations to tensor format."""
        batch_obs = {}
        
        # Get all keys from first observation
        keys = observations[0].keys()
        
        for key in keys:
            # Stack all observations for this key
            values = [obs[key] for obs in observations]
            batch_obs[key] = torch.tensor(np.stack(values), dtype=torch.float32, device=self.device)
        
        return batch_obs
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def end_episode(self, episode_reward: float, episode_length: int, success: bool):
        """Called at end of episode to update statistics."""
        self.episodes_trained += 1
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.success_rate.append(1.0 if success else 0.0)
        
        # Reset total reward for next episode
        self.total_reward = 0.0
    
    def get_statistics(self) -> Dict[str, float]:
        """Get training statistics."""
        stats = {
            'episodes_trained': self.episodes_trained,
            'steps_done': self.steps_done,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'mean_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'mean_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            'success_rate': np.mean(self.success_rate) if self.success_rate else 0.0,
            'recent_loss': np.mean(self.losses[-100:]) if self.losses else 0.0
        }
        return stats
    
    def save_model(self, path: str):
        """Save model weights and training state."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episodes_trained': self.episodes_trained,
            'epsilon': self.epsilon,
            'losses': self.losses
        }, path)
        print(f"💾 Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights and training state."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.episodes_trained = checkpoint['episodes_trained']
        self.epsilon = checkpoint['epsilon']
        self.losses = checkpoint['losses']
        
        print(f"📂 Model loaded from {path}")
        print(f"   Episodes trained: {self.episodes_trained}")
        print(f"   Steps done: {self.steps_done}")
        print(f"   Current epsilon: {self.epsilon:.4f}")


# Utility functions for RL training
def create_rl_drone(observation_space, action_space, config: Optional[Dict] = None) -> RLFirefighterDrone:
    """Factory function to create RL drone with default or custom configuration."""
    default_config = {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 10000,
        'buffer_size': 100000,
        'batch_size': 64,
        'target_update_freq': 1000
    }
    
    if config:
        default_config.update(config)
    
    return RLFirefighterDrone(observation_space, action_space, **default_config)


def evaluate_drone(drone: RLFirefighterDrone, env, num_episodes: int = 10) -> Dict[str, float]:
    """Evaluate trained drone performance."""
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(num_episodes):
        observation, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action = drone.select_action(observation, training=False)  # No exploration
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                if info.get('is_successful', False):
                    success_count += 1
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': success_count / num_episodes,
        'total_episodes': num_episodes
    }