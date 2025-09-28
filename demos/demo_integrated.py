#!/usr/bin/env python3
"""
PyroGuard AI Demo with RL Drone Integration

This demo shows the wildfire environment with a trained RL drone.
It can fall back to the simple drone if no trained model is available.

Usage:
    python demo_rl.py [--model-path path/to/model.pth] [--grid-size 20]
"""

import argparse
import os
import numpy as np
import pygame
import time
import math
from typing import Optional, Dict, Any

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import both environments and drone types
from environment.wildfire_env import WildfireEnvironment
from drones.rl.rl_environment import RLWildfireEnvironment
from drones.rl.rl_agent import create_rl_drone
from drones.simple.firefighter_drone import FirefighterDrone
# from drones.llm.cirrascale_llm_drone import create_cirrascale_llm_drone  # Removed cirrascale support


class RLIntegratedWildfireEnvironment(WildfireEnvironment):
    """
    Enhanced wildfire environment that can use either simple rule-based drone
    or trained RL drone for firefighting operations.
    """
    
    def __init__(self, grid_size: int = 25, 
                 fire_spread_prob: float = 0.3, 
                 initial_tree_density: float = 0.75,
                 wind_strength: float = 0.15,
                 fire_persistence: int = 4,
                 new_fire_rate: float = 0.08,
                 drone_type: str = "simple",
                 rl_model_path: Optional[str] = None):
        """
        Initialize the enhanced wildfire environment.
        
        Args:
            drone_type: "simple" or "rl" for drone type
            rl_model_path: Path to trained RL model (required if drone_type="rl")
            Other args same as base WildfireEnvironment
        """
        
        self.drone_type = drone_type
        self.rl_model_path = rl_model_path
        
        # Initialize base environment with drones enabled so rendering works
        super().__init__(
            grid_size=grid_size,
            fire_spread_prob=fire_spread_prob,
            initial_tree_density=initial_tree_density,
            wind_strength=wind_strength,
            fire_persistence=fire_persistence,
            new_fire_rate=new_fire_rate,
            enable_drones=True  # Enable so base class render method works
        )
        
        # Override the base class drone with our custom drone system
        self._initialize_drone_system()
        
        # RL-specific state tracking
        if self.drone_type == "rl":
            self.rl_observation = None
            self.rl_step_count = 0
    
    def _initialize_drone_system(self):
        """Initialize the appropriate drone system."""
        if self.drone_type == "simple":
            # Use simple rule-based drone - replace the base class drone
            start_position = (1, 1)
            self.drone = FirefighterDrone(start_position, self.grid_size)
            print("🚁 Simple rule-based firefighter drone deployed")
            
        elif self.drone_type == "rl":
            if not self.rl_model_path or not os.path.exists(self.rl_model_path):
                print(f"⚠️ RL model not found at {self.rl_model_path}")
                print("🔄 Falling back to simple drone...")
                self.drone_type = "simple"
                self._initialize_drone_system()
                return
            
            # Initialize RL environment wrapper for observations
            self.rl_env_wrapper = RLWildfireEnvironment(
                grid_size=self.grid_size,
                max_episode_steps=1000,
                fire_spawn_rate=self.new_fire_rate
            )
            
            # Create and load trained RL drone
            self.rl_drone = create_rl_drone(
                self.rl_env_wrapper.observation_space,
                self.rl_env_wrapper.action_space
            )
            self.rl_drone.load_model(self.rl_model_path)
            
            # RL drone state
            self.drone_position = [1, 1]
            self.drone_water = 100.0
            self.drone_energy = 100.0
            self.max_water = 100.0
            self.max_energy = 100.0
            self.fires_extinguished = 0
            
            print("🤖 RL-trained firefighter drone deployed")
            print(f"   Model: {self.rl_model_path}")
            print(f"   Training episodes: {self.rl_drone.episodes_trained}")
        
        else:
            raise ValueError(f"Unknown drone type: {self.drone_type}")
    
    def reset(self):
        """Reset environment and drone system."""
        # Store drone_type before calling super().reset() which might overwrite it
        current_drone_type = self.drone_type
        state = super().reset()
        self.drone_type = current_drone_type  # Restore drone_type
        
        if self.drone_type == "simple":
            # Reset simple drone
            if hasattr(self, 'drone'):
                start_position = (1, 1)
                self.drone = FirefighterDrone(start_position, self.grid_size)
        
        elif self.drone_type == "rl":
            # Ensure RL drone attributes are initialized
            if not hasattr(self, 'max_water'):
                self.max_water = 100.0
                self.max_energy = 100.0
            
            # Reset RL drone state
            self.drone_position = [1, 1]
            self.drone_water = self.max_water
            self.drone_energy = self.max_energy
            self.fires_extinguished = 0
            self.rl_step_count = 0
            
            # Sync RL wrapper with current environment state if it exists
            if hasattr(self, 'rl_env_wrapper'):
                self.rl_env_wrapper.env.grid = self.grid.copy()
                self.rl_env_wrapper.env.fire_age = self.fire_age.copy()
                self.rl_env_wrapper.drone_position = self.drone_position.copy()
                self.rl_env_wrapper.drone_water = self.drone_water
                self.rl_env_wrapper.drone_energy = self.drone_energy
        
        return state
    
    def step(self):
        """Step environment with appropriate drone control."""
        # Store previous state for RL drone
        if self.drone_type == "rl":
            prev_observation = self._get_rl_observation()
        
        # Step base wildfire simulation
        state = super().step()
        
        # Update drone system
        if self.drone_type == "simple":
            # Use simple drone logic
            if hasattr(self, 'drone') and self.drone:
                drone_action = self.drone.update(self.grid)
                self._process_simple_drone_action(drone_action)
                
                # Add drone info to state
                state['drone_action'] = drone_action
                state['drone_status'] = self.drone.get_status()
        
        elif self.drone_type == "rl":
            # Use RL drone logic
            current_observation = self._get_rl_observation()
            
            if prev_observation is not None:
                # Get action from RL drone
                action = self.rl_drone.select_action(current_observation, training=False)
                
                # Execute RL drone action
                drone_action = self._execute_rl_drone_action(action)
                
                # Add drone info to state
                state['drone_action'] = drone_action
                state['drone_status'] = self._get_rl_drone_status()
            
            self.rl_step_count += 1
        
        return state
    
    def _get_rl_observation(self) -> Dict[str, np.ndarray]:
        """Get observation for RL drone (mirrors RL environment wrapper)."""
        # Local terrain view around drone
        local_terrain = self._get_local_view(self.grid, self.drone_position)
        local_fire_age = self._get_local_view(self.fire_age, self.drone_position)
        
        # Drone state (normalized)
        drone_state = np.array([
            self.drone_position[0] / self.grid_size,
            self.drone_position[1] / self.grid_size,
            self.drone_water / self.max_water,
            self.drone_energy / self.max_energy,
            min(self._count_fires_in_radius(6) / 10.0, 1.0),
            min(self.rl_step_count / 1000.0, 1.0)
        ], dtype=np.float32)
        
        # Global state
        total_cells = self.grid_size * self.grid_size
        fire_ratio = np.sum((self.grid == 2) | (self.grid == 5)) / total_cells
        vegetation_ratio = np.sum((self.grid == 1) | (self.grid == 6) | (self.grid == 7) | (self.grid == 9)) / total_cells
        
        wind_rad = np.radians(self.wind_direction)
        wind_x = np.cos(wind_rad) * self.wind_strength
        wind_y = np.sin(wind_rad) * self.wind_strength
        fire_spread_intensity = min(fire_ratio * self.wind_strength * 10, 1.0)
        
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
    
    def _get_local_view(self, grid: np.ndarray, position: list, view_size: int = 11) -> np.ndarray:
        """Get local view around given position."""
        half_size = view_size // 2
        local_view = np.zeros((view_size, view_size), dtype=np.int32)
        
        drone_row, drone_col = position
        
        for i in range(view_size):
            for j in range(view_size):
                world_row = drone_row - half_size + i
                world_col = drone_col - half_size + j
                
                if (0 <= world_row < self.grid_size and 0 <= world_col < self.grid_size):
                    local_view[i, j] = grid[world_row, world_col]
                else:
                    local_view[i, j] = -1  # Out of bounds
        
        return local_view
    
    def _count_fires_in_radius(self, radius: int) -> int:
        """Count fires within radius of RL drone."""
        count = 0
        drone_row, drone_col = self.drone_position
        
        for row in range(max(0, drone_row - radius), min(self.grid_size, drone_row + radius + 1)):
            for col in range(max(0, drone_col - radius), min(self.grid_size, drone_col + radius + 1)):
                distance = np.sqrt((row - drone_row)**2 + (col - drone_col)**2)
                if distance <= radius and self.grid[row, col] in [2, 5]:
                    count += 1
        
        return count
    
    def _execute_rl_drone_action(self, action: list) -> Dict[str, Any]:
        """Execute RL drone action and return action info."""
        move_direction, action_type = action
        
        # Movement directions mapping
        movement_directions = {
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
        
        action_info = {
            'action_type': 'rl_control',
            'move_direction': move_direction,
            'action_type_id': action_type,
            'position': tuple(self.drone_position)
        }
        
        # Execute movement
        if move_direction in movement_directions:
            dr, dc = movement_directions[move_direction]
            new_row = self.drone_position[0] + dr
            new_col = self.drone_position[1] + dc
            
            if (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
                self.drone_position = [new_row, new_col]
                self.drone_energy -= 2
                action_info['moved'] = True
            else:
                action_info['moved'] = False
        
        # Execute special action
        if action_type == 1:  # Fire suppression
            suppression_result = self._execute_rl_fire_suppression()
            action_info.update(suppression_result)
        elif action_type == 2:  # Return to base
            base_result = self._execute_rl_return_to_base()
            action_info.update(base_result)
        
        # Ensure resources don't go negative
        self.drone_water = max(0, self.drone_water)
        self.drone_energy = max(0, self.drone_energy)
        
        return action_info
    
    def _execute_rl_fire_suppression(self) -> Dict[str, Any]:
        """Execute fire suppression for RL drone."""
        if self.drone_water <= 0:
            return {'suppression_result': 'no_water'}
        
        row, col = self.drone_position
        fires_suppressed = 0
        
        # Check for fires in suppression range (1 cell)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                target_row, target_col = row + dr, col + dc
                
                if not (0 <= target_row < self.grid_size and 0 <= target_col < self.grid_size):
                    continue
                
                cell_type = self.grid[target_row, target_col]
                if cell_type in [2, 5]:  # Fire or intense fire
                    # Suppress the fire
                    self.grid[target_row, target_col] = 3  # Burned
                    self.fire_age[target_row, target_col] = 0
                    
                    # Resource costs
                    water_cost = 8 if cell_type == 5 else 5
                    self.drone_water -= water_cost
                    self.drone_energy -= 3
                    
                    fires_suppressed += 1
                    self.fires_extinguished += 1
        
        return {
            'suppression_result': 'success' if fires_suppressed > 0 else 'no_fire_in_range',
            'fires_suppressed': fires_suppressed
        }
    
    def _execute_rl_return_to_base(self) -> Dict[str, Any]:
        """Execute return to base for RL drone."""
        base_position = [0, 0]
        distance = np.sqrt((self.drone_position[0] - base_position[0])**2 + 
                          (self.drone_position[1] - base_position[1])**2)
        
        if distance <= 1.5:  # At base
            self.drone_water = self.max_water
            self.drone_energy = self.max_energy
            return {'base_action': 'refueled'}
        else:
            # Move towards base
            direction = [
                base_position[0] - self.drone_position[0],
                base_position[1] - self.drone_position[1]
            ]
            
            if abs(direction[0]) > abs(direction[1]):
                move = [1 if direction[0] > 0 else -1, 0]
            else:
                move = [0, 1 if direction[1] > 0 else -1]
            
            new_pos = [self.drone_position[0] + move[0], 
                      self.drone_position[1] + move[1]]
            
            if (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
                self.drone_position = new_pos
                self.drone_energy -= 1
            
            return {'base_action': 'moving_to_base', 'distance': distance}
    
    def _get_rl_drone_status(self) -> Dict[str, Any]:
        """Get RL drone status for display."""
        return {
            'position': tuple(self.drone_position),
            'water_level': self.drone_water,
            'water_percentage': (self.drone_water / self.max_water) * 100,
            'energy': self.drone_energy,
            'energy_percentage': (self.drone_energy / self.max_energy) * 100,
            'fires_extinguished': self.fires_extinguished,
            'known_fires': self._count_fires_in_radius(6),
            'target_fire': None,  # RL drone doesn't expose this directly
            'steps_taken': self.rl_step_count
        }
    
    def _process_simple_drone_action(self, drone_action: Dict):
        """Process action from simple drone (inherited from parent class)."""
        if drone_action.get('action') == 'fire_suppressed':
            pos = drone_action.get('position')
            if pos and self._is_valid_position(pos):
                if self.grid[pos[0], pos[1]] in [self.FIRE, self.FIRE_INTENSE]:
                    self.grid[pos[0], pos[1]] = self.BURNED
                    self.fire_age[pos[0], pos[1]] = 0
                    self.fires_extinguished += 1
    
    
    def render(self, mode='human'):
        """Enhanced rendering - base class handles drone rendering."""
        # For RL drone, we need to create a mock drone object for visualization
        if self.drone_type == "rl" and hasattr(self, 'drone_position'):
            self._create_rl_drone_mock()
        
        # Use base class rendering which will draw the drone
        return super().render(mode)
    
    def _create_rl_drone_mock(self):
        """Create a mock drone object for RL drone visualization."""
        class MockDrone:
            def __init__(self, position, water, energy, fires_extinguished):
                self.position = position
                self.water_level = water
                self.energy = energy
                self.max_water = 100.0
                self.max_energy = 100.0
                self.fires_extinguished = fires_extinguished
                self.detection_radius = 6
                self.target_fire = None  # RL drone doesn't expose this
                self.trail = []  # Could track this if needed
            
            def update(self, grid):
                """Mock update method - RL drone logic is handled separately."""
                return {
                    'action': 'rl_controlled',
                    'position': tuple(self.position),
                    'water_used': 0,
                    'energy_used': 0
                }
                
            def get_status(self):
                return {
                    'position': tuple(self.position),
                    'water_level': self.water_level,
                    'water_percentage': (self.water_level / self.max_water) * 100,
                    'energy': self.energy,
                    'energy_percentage': (self.energy / self.max_energy) * 100,
                    'fires_extinguished': self.fires_extinguished,
                    'known_fires': 0,  # Could calculate this if needed
                    'target_fire': self.target_fire,
                    'steps_taken': 0
                }
        
        # Create mock drone with current RL drone state
        self.drone = MockDrone(
            position=self.drone_position,
            water=self.drone_water,
            energy=self.drone_energy,
            fires_extinguished=self.fires_extinguished
        )
        """Draw RL drone with distinct visual style."""
        if not hasattr(self, 'screen') or self.screen is None:
            return
        
        row, col = self.drone_position
        
        # Calculate screen position
        x = col * self.cell_size + 20
        y = row * self.cell_size + 20
        
        # Get elevation for 3D positioning
        elevation = self._get_elevation(row, col)
        height_offset = int(elevation * self.elevation_scale) + 20  # RL drone flies higher
        
        screen_x = x + self.cell_size // 2
        screen_y = y + self.cell_size // 2 - height_offset
        
        # RL drone color (different from simple drone)
        base_color = (100, 255, 100)  # Green for RL
        drone_size = 14  # Slightly larger
        
        # Modify color based on status
        if self.drone_water < 20:
            state_color = (255, 255, 0)  # Yellow when low on water
        elif self.drone_energy < 30:
            state_color = (255, 150, 0)  # Orange when low energy
        else:
            state_color = base_color
        
        # Draw drone shadow
        shadow_pos = (screen_x + 3, screen_y + 18)
        pygame.draw.circle(self.screen, (30, 30, 30), shadow_pos, drone_size // 2)
        
        # Draw main drone body
        pygame.draw.circle(self.screen, (60, 60, 60), (screen_x, screen_y + 2), drone_size)
        pygame.draw.circle(self.screen, state_color, (screen_x, screen_y), drone_size)
        pygame.draw.circle(self.screen, (255, 255, 255), (screen_x - 3, screen_y - 3), drone_size // 3)
        
        # Draw AI indicator
        pygame.draw.circle(self.screen, (0, 255, 255), (screen_x, screen_y), drone_size // 4)
        
        # Draw drone ID
        if hasattr(self, 'screen'):
            font = pygame.font.Font(None, 18)
            id_text = font.render("AI", True, (0, 255, 255))
            self.screen.blit(id_text, (screen_x - 8, screen_y + drone_size + 8))


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='PyroGuard AI Demo with RL Drone')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained RL model')
    parser.add_argument('--grid-size', type=int, default=25,
                       help='Grid size (default: 25)')
    parser.add_argument('--drone-type', type=str, choices=['simple', 'rl'], default='simple',
                       help='Drone type to use (default: simple)')
    
    args = parser.parse_args()
    
    # Determine drone type and prepare configurations
    drone_type = args.drone_type
    
    if drone_type == "rl" and not args.model_path:
        print("⚠️ RL drone requested but no model path provided")
        print("🔍 Looking for trained models in 'models' directory...")
        
        # Look for available models
        if os.path.exists('models'):
            model_files = [f for f in os.listdir('models') if f.endswith('.pth')]
            if model_files:
                # Use the most recent model
                model_files.sort(reverse=True)
                args.model_path = os.path.join('models', model_files[0])
                print(f"📂 Found model: {args.model_path}")
            else:
                print("❌ No trained models found, falling back to simple drone")
                drone_type = "simple"
        else:
            print("❌ Models directory not found, falling back to simple drone")
            drone_type = "simple"
    
    
    print(f"🔥 PyroGuard AI Demo - {drone_type.upper()} Drone")
    print("=" * 50)
    
    # Create enhanced environment
    env = RLIntegratedWildfireEnvironment(
        grid_size=args.grid_size,
        fire_spread_prob=0.25,
        initial_tree_density=0.75,
        wind_strength=0.12,
        fire_persistence=5,
        new_fire_rate=0.02,
        drone_type=drone_type,
        rl_model_path=args.model_path
    )
    
    # Initialize and run
    state = env.reset()
    print(f"🌲 Environment initialized with {drone_type} drone")
    print("🎮 Controls:")
    print("  - SPACEBAR: Spawn fires")
    print("  - W: Change wind")
    print("  - R: Reset environment")
    print("  - Click: Ignite terrain")
    print("  - Close window or Ctrl+C to exit")
    
    try:
        step = 0
        while True:
            if not env.render():
                break
            
            state = env.step()
            step += 1
            
            # Print status updates
            if step % 100 == 0:
                drone_status = state.get('drone_status', {})
                fires_active = state.get('fires_active', 0)
                fires_extinguished = drone_status.get('fires_extinguished', 0)
                
                print(f"Step {step}: {fires_active} fires active, "
                      f"{fires_extinguished} extinguished by {drone_type} drone")
    
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    finally:
        env.close()
        print("🔥 Thanks for trying PyroGuard AI!")


if __name__ == "__main__":
    main()