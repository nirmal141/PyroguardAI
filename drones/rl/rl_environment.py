import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from environment.wildfire_env import WildfireEnvironment


class RLWildfireEnvironment(gym.Env):
    """
    Gymnasium-compatible RL environment for training firefighter drones.
    
    This environment wraps the WildfireEnvironment to provide:
    - Proper observation and action spaces for RL training
    - Reward shaping for effective learning
    - Multi-objective rewards (fire suppression, efficiency, survival)
    - Episodic training structure
    """
    
    def __init__(self, 
                 grid_size: int = 20,
                 max_episode_steps: int = 500,
                 fire_spawn_rate: float = 0.02,
                 reward_config: Optional[Dict[str, float]] = None):
        """
        Initialize the RL environment.
        
        Args:
            grid_size: Size of the simulation grid
            max_episode_steps: Maximum steps per episode
            fire_spawn_rate: Rate of automatic fire spawning
            reward_config: Dictionary of reward weights
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        # Initialize base wildfire environment (without built-in drones)
        self.env = WildfireEnvironment(
            grid_size=grid_size,
            fire_spread_prob=0.15,
            initial_tree_density=0.7,
            wind_strength=0.1,
            fire_persistence=6,
            new_fire_rate=fire_spawn_rate,
            enable_drones=False  # We control the drone through RL
        )
        
        # Drone state
        self.drone_position = [1, 1]  # [row, col]
        self.drone_water = 100.0
        self.drone_energy = 500.0  # Same as simple drone
        self.max_water = 100.0
        self.max_energy = 500.0  # Same as simple drone
        
        # Drone capabilities
        self.detection_radius = 5  # Can detect fires within 5 cells (same as simple drone)
        self.suppression_range = 1  # Can suppress fires within 1 cell (same as simple drone)
        self.movement_range = 2  # Can move up to 2 cells per action (same as simple drone)
        
        # Performance tracking
        self.fires_extinguished = 0
        self.trees_saved = 0
        self.water_usage_efficiency = 0.0
        
        # Reward configuration
        self.reward_config = reward_config or {
            'fire_suppressed': 10.0,
            'tree_saved': 2.0,
            'water_efficiency': 0.1,
            'energy_efficiency': 0.05,
            'proximity_to_fire': 0.2,
            'coverage_bonus': 0.5,
            'time_penalty': -0.1,
            'crash_penalty': -50.0,
            'episode_success': 100.0
        }
        
        # Define observation space
        # Observation includes:
        # 1. Local grid view around drone (11x11 window)
        # 2. Drone state (position, water, energy)
        # 3. Global fire statistics
        # 4. Wind information
        local_view_size = 11
        self.local_view_size = local_view_size
        
        self.observation_space = spaces.Dict({
            # Local terrain view (11x11 around drone)
            'local_terrain': spaces.Box(
                low=0, high=9, 
                shape=(local_view_size, local_view_size), 
                dtype=np.int32
            ),
            # Local fire intensity view
            'local_fire_age': spaces.Box(
                low=0, high=10,
                shape=(local_view_size, local_view_size),
                dtype=np.int32
            ),
            # Drone state vector
            'drone_state': spaces.Box(
                low=0.0, high=1.0,
                shape=(6,),  # [norm_pos_x, norm_pos_y, water_level, energy_level, fires_detected, step_progress]
                dtype=np.float32
            ),
            # Global environment state
            'global_state': spaces.Box(
                low=0.0, high=1.0,
                shape=(5,),  # [fire_ratio, vegetation_ratio, wind_x, wind_y, fire_spread_intensity]
                dtype=np.float32
            )
        })
        
        # Define action space
        # Actions: [move_direction, action_type]
        # move_direction: 0-8 (stay, N, NE, E, SE, S, SW, W, NW)
        # action_type: 0-2 (move_only, suppress_fire, return_to_base)
        self.action_space = spaces.MultiDiscrete([9, 3])
        
        # Movement directions
        self.movement_directions = {
            0: (0, 0),   # Stay
            1: (-1, 0),  # North
            2: (-1, 1),  # Northeast
            3: (0, 1),   # East
            4: (1, 1),   # Southeast
            5: (1, 0),   # South
            6: (1, -1),  # Southwest
            7: (0, -1),  # West
            8: (-1, -1)  # Northwest
        }
        
        # Initialize environment
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset wildfire environment
        self.env.reset()
        
        # Reset drone state
        self.drone_position = [1, 1]
        self.drone_water = self.max_water
        self.drone_energy = self.max_energy
        
        # Reset tracking variables
        self.current_step = 0
        self.fires_extinguished = 0
        self.trees_saved = 0
        self.water_usage_efficiency = 0.0
        
        # Spawn initial fires for training
        self.env.spawn_random_fires(np.random.randint(2, 5))
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        self.current_step += 1
        
        # Parse action
        move_direction, action_type = action[0], action[1]
        
        # Store previous state for reward calculation
        prev_fires = np.sum((self.env.grid == 2) | (self.env.grid == 5))
        prev_vegetation = np.sum((self.env.grid == 1) | (self.env.grid == 6) | (self.env.grid == 7) | (self.env.grid == 9))
        prev_water = self.drone_water
        prev_energy = self.drone_energy
        
        # Execute drone action
        reward = 0.0
        action_info = {}
        
        # 1. Movement
        if move_direction != 0:  # If not staying in place
            reward += self._execute_movement(move_direction)
        
        # 2. Special actions
        if action_type == 1:  # Suppress fire
            suppress_reward, suppress_info = self._execute_fire_suppression()
            reward += suppress_reward
            action_info.update(suppress_info)
        elif action_type == 2:  # Return to base
            base_reward, base_info = self._execute_return_to_base()
            reward += base_reward
            action_info.update(base_info)
        
        # 3. Step the wildfire environment
        env_state = self.env.step()
        
        # 4. Base energy consumption (same as simple drone)
        self.drone_energy -= 0.5
        self.drone_energy = max(0, self.drone_energy)
        
        # 5. Calculate rewards
        reward += self._calculate_environmental_reward(prev_fires, prev_vegetation, prev_water, prev_energy)
        
        # 5. Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_episode_steps
        
        # Add success bonus if episode ended successfully
        if terminated and self._is_successful_episode():
            reward += self.reward_config['episode_success']
        
        # 6. Spawn occasional new fires during training
        if np.random.random() < 0.01:  # 1% chance per step
            self.env.spawn_random_fires(1)
        
        observation = self._get_observation()
        info = self._get_info()
        info.update(action_info)
        
        return observation, reward, terminated, truncated, info
    
    def _execute_movement(self, direction: int) -> float:
        """Execute movement action and return movement reward."""
        if direction not in self.movement_directions:
            return self.reward_config['crash_penalty']
        
        dr, dc = self.movement_directions[direction]
        
        # Move up to movement_range cells per action (same as simple drone)
        moves_made = 0
        current_row, current_col = self.drone_position
        
        while moves_made < self.movement_range:
            new_row = current_row + dr
            new_col = current_col + dc
            
            # Check bounds
            if not (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
                break  # Stop if we hit a boundary
            
            # Check if moving into dangerous terrain
            cell_type = self.env.grid[new_row, new_col]
            if cell_type == 4:  # Water - drone can fly over it
                pass
            elif cell_type in [2, 5]:  # Fire - dangerous but possible
                self.drone_energy -= 5  # Extra energy cost in fire
            
            # Update position
            current_row, current_col = new_row, new_col
            moves_made += 1
        
        # Update drone position
        self.drone_position = [current_row, current_col]
        
        # Energy cost for movement (proportional to distance moved)
        self.drone_energy -= 1 * moves_made  # Reduced to be more consistent with simple drone
        self.drone_energy = max(0, self.drone_energy)
        
        # Small reward for moving towards fires
        fires_nearby = self._count_fires_in_radius(self.detection_radius)
        proximity_reward = fires_nearby * self.reward_config['proximity_to_fire']
        
        return proximity_reward + self.reward_config['time_penalty']
    
    def _execute_fire_suppression(self) -> Tuple[float, Dict[str, Any]]:
        """Execute fire suppression action."""
        if self.drone_water <= 0:
            return 0.0, {'suppression_result': 'no_water'}
        
        reward = 0.0
        info = {'suppression_result': 'no_fire_in_range', 'fires_suppressed': 0}
        
        # Check for fires in suppression range
        row, col = self.drone_position
        fires_suppressed = 0
        
        for dr in range(-self.suppression_range, self.suppression_range + 1):
            for dc in range(-self.suppression_range, self.suppression_range + 1):
                target_row, target_col = row + dr, col + dc
                
                if not (0 <= target_row < self.grid_size and 0 <= target_col < self.grid_size):
                    continue
                
                cell_type = self.env.grid[target_row, target_col]
                if cell_type in [2, 5]:  # Fire
                    # Suppress the fire
                    self.env.grid[target_row, target_col] = 3  # Burned
                    self.env.fire_age[target_row, target_col] = 0
                    
                    # Resource costs
                    water_cost = 8 if cell_type == 5 else 5  # Intense fires cost more
                    self.drone_water -= water_cost
                    self.drone_water = max(0, self.drone_water)
                    
                    self.drone_energy -= 3
                    self.drone_energy = max(0, self.drone_energy)
                    
                    # Rewards
                    fire_reward = self.reward_config['fire_suppressed']
                    if cell_type == 5:  # Intense fire bonus
                        fire_reward *= 1.5
                    
                    reward += fire_reward
                    fires_suppressed += 1
                    self.fires_extinguished += 1
                    
                    # Save surrounding vegetation
                    self.trees_saved += self._count_vegetation_neighbors(target_row, target_col)
        
        info['suppression_result'] = 'success' if fires_suppressed > 0 else 'no_fire_in_range'
        info['fires_suppressed'] = fires_suppressed
        
        return reward, info
    
    def _execute_return_to_base(self) -> Tuple[float, Dict[str, Any]]:
        """Execute return to base action."""
        base_position = [0, 0]
        distance_to_base = math.sqrt(
            (self.drone_position[0] - base_position[0])**2 + 
            (self.drone_position[1] - base_position[1])**2
        )
        
        if distance_to_base <= 1.5:  # At base
            # Refuel
            water_refilled = self.max_water - self.drone_water
            energy_refilled = self.max_energy - self.drone_energy
            
            self.drone_water = self.max_water
            self.drone_energy = self.max_energy
            
            # Small reward for successful refueling
            refuel_reward = (water_refilled + energy_refilled) * 0.01
            
            return refuel_reward, {
                'base_action': 'refueled',
                'water_refilled': water_refilled,
                'energy_refilled': energy_refilled
            }
        else:
            # Move towards base
            direction_to_base = [
                base_position[0] - self.drone_position[0],
                base_position[1] - self.drone_position[1]
            ]
            
            # Normalize and move
            if abs(direction_to_base[0]) > abs(direction_to_base[1]):
                move_direction = (-1, 0) if direction_to_base[0] < 0 else (1, 0)
            else:
                move_direction = (0, -1) if direction_to_base[1] < 0 else (0, 1)
            
            new_position = [
                self.drone_position[0] + move_direction[0],
                self.drone_position[1] + move_direction[1]
            ]
            
            # Check bounds
            if (0 <= new_position[0] < self.grid_size and 
                0 <= new_position[1] < self.grid_size):
                self.drone_position = new_position
                self.drone_energy -= 1  # Lower energy cost when returning to base
                self.drone_energy = max(0, self.drone_energy)
            
            return 0.0, {'base_action': 'moving_to_base', 'distance': distance_to_base}
    
    def _calculate_environmental_reward(self, prev_fires: int, prev_vegetation: int, 
                                      prev_water: float, prev_energy: float) -> float:
        """Calculate environmental and efficiency rewards."""
        reward = 0.0
        
        # Fire progression penalty
        current_fires = np.sum((self.env.grid == 2) | (self.env.grid == 5))
        if current_fires > prev_fires:
            reward -= (current_fires - prev_fires) * 2.0
        
        # Vegetation loss penalty
        current_vegetation = np.sum((self.env.grid == 1) | (self.env.grid == 6) | (self.env.grid == 7) | (self.env.grid == 9))
        vegetation_lost = prev_vegetation - current_vegetation
        if vegetation_lost > 0:
            reward -= vegetation_lost * 0.5
        
        # Resource efficiency rewards
        water_efficiency = 1.0 - (prev_water - self.drone_water) / max(prev_water, 1.0)
        energy_efficiency = 1.0 - (prev_energy - self.drone_energy) / max(prev_energy, 1.0)
        
        reward += water_efficiency * self.reward_config['water_efficiency']
        reward += energy_efficiency * self.reward_config['energy_efficiency']
        
        # Coverage bonus (encourage exploration)
        if self._is_exploring_new_area():
            reward += self.reward_config['coverage_bonus']
        
        return reward
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation for the RL agent."""
        # Local terrain view
        local_terrain = self._get_local_view(self.env.grid)
        local_fire_age = self._get_local_view(self.env.fire_age)
        
        # Drone state (normalized)
        drone_state = np.array([
            self.drone_position[0] / self.grid_size,  # Normalized row
            self.drone_position[1] / self.grid_size,  # Normalized col
            self.drone_water / self.max_water,        # Water level
            self.drone_energy / self.max_energy,      # Energy level
            min(self._count_fires_in_radius(self.detection_radius) / 10.0, 1.0),  # Fires detected
            self.current_step / self.max_episode_steps  # Episode progress
        ], dtype=np.float32)
        
        # Global state
        total_cells = self.grid_size * self.grid_size
        fire_ratio = np.sum((self.env.grid == 2) | (self.env.grid == 5)) / total_cells
        vegetation_ratio = np.sum((self.env.grid == 1) | (self.env.grid == 6) | (self.env.grid == 7) | (self.env.grid == 9)) / total_cells
        
        # Wind components (normalized)
        wind_rad = np.radians(self.env.wind_direction)
        wind_x = np.cos(wind_rad) * self.env.wind_strength
        wind_y = np.sin(wind_rad) * self.env.wind_strength
        
        fire_spread_intensity = min(fire_ratio * self.env.wind_strength * 10, 1.0)
        
        global_state = np.array([
            fire_ratio,
            vegetation_ratio,
            wind_x,
            wind_y,
            fire_spread_intensity
        ], dtype=np.float32)
        
        return {
            'local_terrain': local_terrain,
            'local_fire_age': local_fire_age,
            'drone_state': drone_state,
            'global_state': global_state
        }
    
    def _get_local_view(self, grid: np.ndarray) -> np.ndarray:
        """Get local view around drone position."""
        half_size = self.local_view_size // 2
        local_view = np.zeros((self.local_view_size, self.local_view_size), dtype=np.int32)
        
        drone_row, drone_col = self.drone_position
        
        for i in range(self.local_view_size):
            for j in range(self.local_view_size):
                world_row = drone_row - half_size + i
                world_col = drone_col - half_size + j
                
                if (0 <= world_row < self.grid_size and 0 <= world_col < self.grid_size):
                    local_view[i, j] = grid[world_row, world_col]
                else:
                    local_view[i, j] = -1  # Out of bounds marker
        
        return local_view
    
    def _count_fires_in_radius(self, radius: int) -> int:
        """Count fires within given radius of drone."""
        count = 0
        drone_row, drone_col = self.drone_position
        
        for row in range(max(0, drone_row - radius), min(self.grid_size, drone_row + radius + 1)):
            for col in range(max(0, drone_col - radius), min(self.grid_size, drone_col + radius + 1)):
                distance = math.sqrt((row - drone_row)**2 + (col - drone_col)**2)
                if distance <= radius and self.env.grid[row, col] in [2, 5]:
                    count += 1
        
        return count
    
    def _count_vegetation_neighbors(self, row: int, col: int) -> int:
        """Count vegetation cells neighboring given position."""
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if (0 <= nr < self.grid_size and 0 <= nc < self.grid_size and
                    self.env.grid[nr, nc] in [1, 6, 7, 9]):
                    count += 1
        return count
    
    def _is_exploring_new_area(self) -> bool:
        """Check if drone is exploring a new area (simplified)."""
        # This is a simplified implementation
        # In practice, you might want to track visited cells
        return np.random.random() < 0.1  # 10% chance to encourage exploration
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if drone is out of energy and not at base
        if self.drone_energy <= 0:
            distance_to_base = math.sqrt(self.drone_position[0]**2 + self.drone_position[1]**2)
            if distance_to_base > 1.5:
                return True
        
        # Terminate if no fires left (success)
        fires_remaining = np.sum((self.env.grid == 2) | (self.env.grid == 5))
        if fires_remaining == 0:
            return True
        
        # Terminate if too much vegetation burned (failure)
        total_vegetation = np.sum((self.env.grid == 1) | (self.env.grid == 6) | (self.env.grid == 7) | (self.env.grid == 9))
        burned_vegetation = np.sum(self.env.grid == 3)
        
        if total_vegetation + burned_vegetation > 0:
            burn_ratio = burned_vegetation / (total_vegetation + burned_vegetation)
            if burn_ratio > 0.8:  # 80% burned - mission failed
                return True
        
        return False
    
    def _is_successful_episode(self) -> bool:
        """Check if episode ended successfully."""
        fires_remaining = np.sum((self.env.grid == 2) | (self.env.grid == 5))
        return fires_remaining == 0
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state."""
        fires_active = np.sum((self.env.grid == 2) | (self.env.grid == 5))
        vegetation_remaining = np.sum((self.env.grid == 1) | (self.env.grid == 6) | (self.env.grid == 7) | (self.env.grid == 9))
        
        return {
            'step': self.current_step,
            'drone_position': self.drone_position.copy(),
            'drone_water': self.drone_water,
            'drone_energy': self.drone_energy,
            'fires_active': fires_active,
            'fires_extinguished': self.fires_extinguished,
            'vegetation_remaining': vegetation_remaining,
            'trees_saved': self.trees_saved,
            'is_successful': self._is_successful_episode()
        }
    
    def render(self, mode='human'):
        """Render the environment (delegate to wildfire environment)."""
        # Add drone position to visualization
        # This is a simplified version - you might want to modify wildfire_env.py
        # to show the RL drone position
        return self.env.render(mode)
    
    def close(self):
        """Close the environment."""
        self.env.close()