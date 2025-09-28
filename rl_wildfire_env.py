#!/usr/bin/env python3
"""
Reinforcement Learning Environment for PyroGuard AI
Wraps the wildfire environment for RL training with focus on fire spread control.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
import math
from wildfire_env import WildfireEnvironment
from simple_drone import FirefighterDrone


class RLWildfireEnv(gym.Env):
    """
    RL Environment for firefighting drone with focus on controlling fire spread.
    
    State Space: 5x5 grid around drone + drone status
    Action Space: Move (4 directions) + Scan + Suppress
    Reward Function: Prioritizes extinguishing fires that can cause widest spread
    """
    
    def __init__(self, grid_size: int = 30, max_steps: int = 1000):
        super().__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0
        
        # Initialize the base wildfire environment
        self.base_env = WildfireEnvironment(
            grid_size=grid_size,
            fire_spread_prob=0.1,
            initial_tree_density=0.75,
            wind_strength=0.05,
            fire_persistence=8,
            new_fire_rate=0.01,
            enable_drones=True
        )
        
        # State space: 5x5 grid around drone + drone status (6 values)
        # 5x5 = 25 cells, each cell can be: empty, tree, fire, burned, water, rock, etc.
        # Drone status: water_level, energy, fires_extinguished, known_fires_count, refueling, distance_to_nearest_fire
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(31,), dtype=np.float32  # 25 (5x5) + 6 (status)
        )
        
        # Action space: 6 actions
        # 0: Move North, 1: Move South, 2: Move East, 3: Move West
        # 4: Scan for fires, 5: Suppress nearby fires
        self.action_space = spaces.Discrete(6)
        
        # Reward tracking
        self.previous_fires_extinguished = 0
        self.previous_fires_active = 0
        self.previous_total_fires = 0
        
        # Fire spread potential tracking
        self.fire_spread_potential = {}
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment and return initial observation."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.previous_fires_extinguished = 0
        self.previous_fires_active = 0
        self.previous_total_fires = 0
        self.fire_spread_potential = {}
        
        # Reset base environment
        state = self.base_env.reset()
        
        # Spawn some initial fires for training
        self.base_env.spawn_random_fires(3)
        
        # Get initial observation
        observation = self._get_observation()
        
        return observation, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.current_step += 1
        
        # Execute action
        reward = self._execute_action(action)
        
        # Step the base environment
        state = self.base_env.step()
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        total_reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        
        # Update tracking variables
        self.previous_fires_extinguished = self.base_env.drone.fires_extinguished
        self.previous_fires_active = state['fires_active']
        
        info = {
            'fires_active': state['fires_active'],
            'fires_extinguished': self.base_env.drone.fires_extinguished,
            'drone_energy': self.base_env.drone.energy,
            'drone_water': self.base_env.drone.water_level,
            'step': self.current_step
        }
        
        return observation, total_reward, terminated, truncated, info
    
    def _execute_action(self, action: int) -> float:
        """Execute the given action and return immediate reward."""
        drone = self.base_env.drone
        
        if action == 0:  # Move North
            new_pos = (drone.position[0] - 1, drone.position[1])
            if self._is_valid_position(new_pos):
                drone.position = list(new_pos)
                return 0.1  # Small reward for movement
        elif action == 1:  # Move South
            new_pos = (drone.position[0] + 1, drone.position[1])
            if self._is_valid_position(new_pos):
                drone.position = list(new_pos)
                return 0.1
        elif action == 2:  # Move East
            new_pos = (drone.position[0], drone.position[1] + 1)
            if self._is_valid_position(new_pos):
                drone.position = list(new_pos)
                return 0.1
        elif action == 3:  # Move West
            new_pos = (drone.position[0], drone.position[1] - 1)
            if self._is_valid_position(new_pos):
                drone.position = list(new_pos)
                return 0.1
        elif action == 4:  # Scan for fires
            # Use drone's built-in scanning
            detected_fires = drone._scan_for_fires(self.base_env.grid)
            drone._update_known_fires(detected_fires, self.base_env.grid)
            return 0.2  # Reward for gathering information
        elif action == 5:  # Suppress fires
            # Use drone's built-in suppression
            result = drone._try_suppress_fire(self.base_env.grid)
            if result['action'] == 'fire_suppressed':
                return 5.0  # High reward for fire suppression
            else:
                return -0.1  # Small penalty for failed suppression
        
        return 0.0
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation (5x5 grid + drone status)."""
        drone = self.base_env.drone
        row, col = drone.position
        
        # Extract 5x5 grid around drone
        grid_obs = np.zeros((5, 5), dtype=np.float32)
        for i in range(5):
            for j in range(5):
                obs_row = row - 2 + i
                obs_col = col - 2 + j
                
                if 0 <= obs_row < self.grid_size and 0 <= obs_col < self.grid_size:
                    grid_obs[i, j] = float(self.base_env.grid[obs_row, obs_col])
                else:
                    grid_obs[i, j] = -1.0  # Out of bounds
        
        # Flatten grid
        grid_flat = grid_obs.flatten()
        
        # Drone status features
        status_features = np.array([
            drone.water_level / drone.max_water,  # Normalized water level
            drone.energy / drone.max_energy,       # Normalized energy
            min(drone.fires_extinguished / 10.0, 1.0),  # Normalized fires extinguished
            min(len(drone.known_fires) / 5.0, 1.0),     # Normalized known fires count
            1.0 if drone.refueling_time > 0 else 0.0,   # Refueling status
            self._get_distance_to_nearest_fire() / 10.0  # Normalized distance to nearest fire
        ], dtype=np.float32)
        
        # Combine grid and status
        observation = np.concatenate([grid_flat, status_features])
        
        return observation
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on fire spread control objectives."""
        drone = self.base_env.drone
        current_fires = np.sum((self.base_env.grid == self.base_env.FIRE) | 
                             (self.base_env.grid == self.base_env.FIRE_INTENSE))
        
        reward = 0.0
        
        # 1. Fire suppression reward (highest priority)
        fires_extinguished = drone.fires_extinguished - self.previous_fires_extinguished
        if fires_extinguished > 0:
            # Calculate spread potential of extinguished fires
            spread_bonus = self._calculate_fire_spread_potential_bonus()
            reward += 10.0 * fires_extinguished + spread_bonus
        
        # 2. Fire spread prevention reward
        fire_spread_penalty = self._calculate_fire_spread_penalty()
        reward += fire_spread_penalty
        
        # 3. Efficiency rewards
        if drone.target_fire:
            distance_to_target = self._distance_to(drone.target_fire)
            if distance_to_target < 3:
                reward += 1.0  # Reward for being close to target fire
        
        # 4. Resource management
        if drone.energy < 20 or drone.water_level == 0:
            reward -= 2.0  # Penalty for poor resource management
        
        # 5. Base camping penalty (NEW - forces drone to leave base ASAP)
        if drone.position == (0, 0) and drone.refueling_time == 0:
            # Drone is at base but not refueling - apply penalty
            reward -= 3.0  # Strong penalty for unnecessary base camping
            # Additional penalty if there are active fires
            if current_fires > 0:
                reward -= 2.0  # Extra penalty when fires are burning
        
        # 6. Exploration bonus for discovering new fires
        if len(drone.known_fires) > 0:
            reward += 0.5
        
        # 7. Exploration efficiency bonus (NEW)
        exploration_bonus = self._calculate_exploration_bonus()
        reward += exploration_bonus
        
        # 8. Fire extinguishing efficiency bonus (NEW)
        efficiency_bonus = self._calculate_efficiency_bonus()
        reward += efficiency_bonus
        
        # 9. Distance-based exploration bonus (NEW)
        distance_bonus = self._calculate_distance_bonus()
        reward += distance_bonus
        
        # 10. Time penalty (encourage efficiency)
        reward -= 0.1
        
        return reward
    
    def _calculate_fire_spread_potential_bonus(self) -> float:
        """Calculate bonus reward for extinguishing fires with high spread potential."""
        bonus = 0.0
        
        # Find fires that were recently extinguished
        current_fires = np.where((self.base_env.grid == self.base_env.FIRE) | 
                                (self.base_env.grid == self.base_env.FIRE_INTENSE))
        current_fire_positions = set(zip(current_fires[0], current_fires[1]))
        
        # Check if any high-spread-potential fires were extinguished
        for fire_pos in list(self.fire_spread_potential.keys()):
            if fire_pos not in current_fire_positions:
                # This fire was extinguished
                potential = self.fire_spread_potential[fire_pos]
                bonus += potential * 2.0  # Bonus proportional to spread potential
                del self.fire_spread_potential[fire_pos]
        
        return bonus
    
    def _calculate_fire_spread_penalty(self) -> float:
        """Calculate penalty for fires spreading."""
        penalty = 0.0
        
        # Update fire spread potential for all current fires
        current_fires = np.where((self.base_env.grid == self.base_env.FIRE) | 
                                (self.base_env.grid == self.base_env.FIRE_INTENSE))
        
        for fire_row, fire_col in zip(current_fires[0], current_fires[1]):
            fire_pos = (fire_row, fire_col)
            
            # Calculate spread potential (number of flammable neighbors)
            spread_potential = self._calculate_spread_potential(fire_row, fire_col)
            self.fire_spread_potential[fire_pos] = spread_potential
            
            # Penalty for fires with high spread potential
            if spread_potential > 3:
                penalty -= spread_potential * 0.5
        
        return penalty
    
    def _calculate_spread_potential(self, row: int, col: int) -> int:
        """Calculate how many flammable neighbors a fire has."""
        neighbors = self.base_env.get_neighbors(row, col)
        flammable_count = 0
        
        for neighbor_row, neighbor_col in neighbors:
            cell_type = self.base_env.grid[neighbor_row, neighbor_col]
            if cell_type in [self.base_env.TREE, self.base_env.DENSE_FOREST, 
                           self.base_env.SPARSE_FOREST, self.base_env.GRASS]:
                flammable_count += 1
        
        return flammable_count
    
    def _calculate_exploration_bonus(self) -> float:
        """Calculate bonus for exploring new areas."""
        drone = self.base_env.drone
        
        # Bonus for being far from base (encourages exploration)
        distance_from_base = self._distance_to((0, 0))
        if distance_from_base > 10:
            return 0.3  # Bonus for exploring far from base
        elif distance_from_base > 5:
            return 0.1  # Small bonus for moderate exploration
        
        return 0.0
    
    def _calculate_efficiency_bonus(self) -> float:
        """Calculate bonus for efficient fire extinguishing."""
        drone = self.base_env.drone
        
        # Bonus for extinguishing fires quickly
        if drone.fires_extinguished > 0:
            efficiency = drone.fires_extinguished / max(drone.steps_taken, 1)
            if efficiency > 0.1:  # More than 0.1 fires per step
                return 1.0  # High bonus for efficiency
            elif efficiency > 0.05:  # More than 0.05 fires per step
                return 0.5  # Medium bonus
        
        return 0.0
    
    def _calculate_distance_bonus(self) -> float:
        """Calculate bonus for moving toward fires."""
        drone = self.base_env.drone
        
        if drone.target_fire:
            distance = self._distance_to(drone.target_fire)
            # Bonus for getting closer to target fire
            if distance < 2:
                return 0.5  # Close to fire
            elif distance < 5:
                return 0.2  # Approaching fire
        
        return 0.0
    
    def _get_distance_to_nearest_fire(self) -> float:
        """Get distance to nearest fire."""
        drone = self.base_env.drone
        
        if not drone.known_fires:
            return 10.0  # Large distance if no fires known
        
        min_distance = float('inf')
        for fire_pos in drone.known_fires:
            distance = self._distance_to(fire_pos)
            min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 10.0
    
    def _distance_to(self, target_pos: Tuple[int, int]) -> float:
        """Calculate distance to target position."""
        drone = self.base_env.drone
        return math.sqrt((drone.position[0] - target_pos[0])**2 + 
                        (drone.position[1] - target_pos[1])**2)
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds."""
        row, col = pos
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if no fires left
        current_fires = np.sum((self.base_env.grid == self.base_env.FIRE) | 
                             (self.base_env.grid == self.base_env.FIRE_INTENSE))
        return current_fires == 0
    
    def render(self, mode='human'):
        """Render the environment."""
        return self.base_env.render()
    
    def close(self):
        """Clean up resources."""
        self.base_env.close()


def test_rl_environment():
    """Test the RL environment."""
    print("🧪 Testing RL Wildfire Environment...")
    
    env = RLWildfireEnv(grid_size=20, max_steps=100)
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test random actions
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step}: Action {action}, Reward {reward:.2f}, "
              f"Fires: {info['fires_active']}, Energy: {info['drone_energy']}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("✅ RL Environment test completed!")


if __name__ == "__main__":
    test_rl_environment()
