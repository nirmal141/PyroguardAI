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
from typing import Optional, Dict, Any, List

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


class MultiDroneWildfireEnvironment(WildfireEnvironment):
    """
    Enhanced wildfire environment that supports multiple drones working together.
    Supports combinations of simple rule-based and RL-trained drones.
    """
    
    def __init__(self, grid_size: int = 25, 
                 fire_spread_prob: float = 0.3, 
                 initial_tree_density: float = 0.75,
                 wind_strength: float = 0.15,
                 fire_persistence: int = 4,
                 new_fire_rate: float = 0.08,
                 num_drones: int = 3,
                 drone_types: List[str] = None,
                 rl_model_path: Optional[str] = None):
        """
        Initialize the enhanced multi-drone wildfire environment.
        
        Args:
            num_drones: Number of drones to deploy (default: 3)
            drone_types: List of drone types ["simple", "rl", "simple"] or None for auto
            rl_model_path: Path to trained RL model (used for RL drones)
            Other args same as base WildfireEnvironment
        """
        
        self.num_drones = num_drones
        self.rl_model_path = rl_model_path
        
        # Default drone configuration: 1 RL + 2 simple drones
        if drone_types is None:
            self.drone_types = ["rl"] + ["simple"] * (num_drones - 1)
        else:
            self.drone_types = drone_types[:num_drones]  # Truncate if too many
            
        # Ensure we have the right number of drone types
        while len(self.drone_types) < num_drones:
            self.drone_types.append("simple")
        
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
        
        # Initialize multi-drone system
        self.drones = []
        self.rl_drones_data = []  # For RL-specific tracking
        self._initialize_multi_drone_system()
        
        # Global tracking
        self.total_fires_extinguished = 0
        self.step_count = 0
    
    def _initialize_multi_drone_system(self):
        """Initialize multiple drones of different types."""
        # Starting positions for drones (spread them out)
        start_positions = [
            (1, 1),
            (1, self.grid_size - 2), 
            (self.grid_size - 2, 1),
            (self.grid_size - 2, self.grid_size - 2),
            (self.grid_size // 2, 1),
            (self.grid_size // 2, self.grid_size - 2)
        ]
        
        # Initialize RL environment wrapper (shared by all RL drones)
        self.rl_env_wrapper = None
        rl_model_available = self.rl_model_path and os.path.exists(self.rl_model_path)
        
        if "rl" in self.drone_types and rl_model_available:
            self.rl_env_wrapper = RLWildfireEnvironment(
                grid_size=self.grid_size,
                max_episode_steps=1000,
                fire_spawn_rate=self.new_fire_rate
            )
        
        print(f"🚁 Deploying {self.num_drones} firefighter drones:")
        
        for i in range(self.num_drones):
            drone_type = self.drone_types[i]
            start_pos = start_positions[i % len(start_positions)]
            drone_id = i + 1
            
            if drone_type == "simple":
                drone = FirefighterDrone(start_pos, self.grid_size)
                self.drones.append({
                    'type': 'simple',
                    'drone': drone,
                    'id': drone_id,
                    'position': list(start_pos)
                })
                print(f"   Drone {drone_id}: Simple rule-based at {start_pos}")
                
            elif drone_type == "rl":
                if not rl_model_available:
                    print(f"   ⚠️ RL model not found, falling back to simple for drone {drone_id}")
                    drone = FirefighterDrone(start_pos, self.grid_size)
                    self.drones.append({
                        'type': 'simple',
                        'drone': drone,
                        'id': drone_id,
                        'position': list(start_pos)
                    })
                else:
                    # Create RL drone
                    rl_agent = create_rl_drone(
                        self.rl_env_wrapper.observation_space,
                        self.rl_env_wrapper.action_space
                    )
                    rl_agent.load_model(self.rl_model_path)
                    
                    # RL drone data - Increased resources to match simple drones
                    rl_data = {
                        'agent': rl_agent,
                        'position': list(start_pos),
                        'water': 100.0,
                        'energy': 500.0,
                        'max_water': 100.0,
                        'max_energy': 500.0,
                        'fires_extinguished': 0,
                        'step_count': 0
                    }
                    
                    self.drones.append({
                        'type': 'rl',
                        'drone': None,  # RL drones don't use the simple drone class
                        'id': drone_id,
                        'position': list(start_pos),
                        'rl_data': rl_data
                    })
                    self.rl_drones_data.append(rl_data)
                    print(f"   Drone {drone_id}: RL-trained at {start_pos}")
            
            else:
                raise ValueError(f"Unknown drone type: {drone_type}")
        
        # Set the first drone as the main drone for compatibility with base class rendering
        if self.drones:
            self.drone = self.drones[0].get('drone')  # May be None for RL drones
    
    def reset(self):
        """Reset environment and multi-drone system."""
        state = super().reset()
        
        # Reset global tracking
        self.total_fires_extinguished = 0
        self.step_count = 0
        
        # Only reset drones if they exist (avoid error during initial __init__)
        if hasattr(self, 'drones') and self.drones:
            # Reset all drones
            start_positions = [
                (1, 1),
                (1, self.grid_size - 2), 
                (self.grid_size - 2, 1),
                (self.grid_size - 2, self.grid_size - 2),
                (self.grid_size // 2, 1),
                (self.grid_size // 2, self.grid_size - 2)
            ]
            
            for i, drone_info in enumerate(self.drones):
                start_pos = start_positions[i % len(start_positions)]
                drone_info['position'] = list(start_pos)
                
                if drone_info['type'] == 'simple':
                    # Reset simple drone
                    drone_info['drone'] = FirefighterDrone(start_pos, self.grid_size)
                    
                elif drone_info['type'] == 'rl':
                    # Reset RL drone state
                    rl_data = drone_info['rl_data']
                    rl_data['position'] = list(start_pos)
                    rl_data['water'] = rl_data['max_water']
                    rl_data['energy'] = rl_data['max_energy']
                    rl_data['fires_extinguished'] = 0
                    rl_data['step_count'] = 0
        
        # Sync RL wrapper with current environment state if it exists
        if hasattr(self, 'rl_env_wrapper') and self.rl_env_wrapper:
            self.rl_env_wrapper.env.grid = self.grid.copy()
            self.rl_env_wrapper.env.fire_age = self.fire_age.copy()
        
        return state
    
    def step(self):
        """Step environment with multi-drone control."""
        # Step base wildfire simulation
        state = super().step()
        self.step_count += 1
        
        # Update all drones
        drone_actions = []
        drone_statuses = []
        
        for drone_info in self.drones:
            if drone_info['type'] == 'simple':
                # Simple drone logic
                drone = drone_info['drone']
                if drone:
                    action = drone.update(self.grid)
                    self._process_simple_drone_action(action, drone_info)
                    
                    drone_actions.append({
                        'drone_id': drone_info['id'],
                        'type': 'simple',
                        'action': action
                    })
                    drone_statuses.append({
                        'drone_id': drone_info['id'],
                        'type': 'simple',
                        'status': drone.get_status()
                    })
                    
            elif drone_info['type'] == 'rl':
                # RL drone logic
                rl_data = drone_info['rl_data']
                observation = self._get_rl_observation(drone_info)
                
                # Get action from RL agent
                action = rl_data['agent'].select_action(observation, training=False)
                
                # Execute RL drone action
                rl_action = self._execute_rl_drone_action(action, drone_info)
                
                drone_actions.append({
                    'drone_id': drone_info['id'],
                    'type': 'rl',
                    'action': rl_action
                })
                drone_statuses.append({
                    'drone_id': drone_info['id'],
                    'type': 'rl',
                    'status': self._get_rl_drone_status(drone_info)
                })
                
                rl_data['step_count'] += 1
        
        # Add multi-drone info to state
        state['drone_actions'] = drone_actions
        state['drone_statuses'] = drone_statuses
        state['total_fires_extinguished'] = self.total_fires_extinguished
        state['num_active_drones'] = len(self.drones)
        
        return state
    
    def _get_rl_observation(self, drone_info: Dict) -> Dict[str, np.ndarray]:
        """Get observation for specific RL drone."""
        rl_data = drone_info['rl_data']
        position = rl_data['position']
        
        # Local terrain view around this drone
        local_terrain = self._get_local_view(self.grid, position)
        local_fire_age = self._get_local_view(self.fire_age, position)
        
        # This drone's state (normalized)
        drone_state = np.array([
            position[0] / self.grid_size,
            position[1] / self.grid_size,
            rl_data['water'] / rl_data['max_water'],
            rl_data['energy'] / rl_data['max_energy'],
            min(self._count_fires_in_radius(6, position) / 10.0, 1.0),
            min(rl_data['step_count'] / 1000.0, 1.0)
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
    
    def _count_fires_in_radius(self, radius: int, position: List[int]) -> int:
        """Count fires within radius of given position."""
        count = 0
        drone_row, drone_col = position
        
        for row in range(max(0, drone_row - radius), min(self.grid_size, drone_row + radius + 1)):
            for col in range(max(0, drone_col - radius), min(self.grid_size, drone_col + radius + 1)):
                distance = np.sqrt((row - drone_row)**2 + (col - drone_col)**2)
                if distance <= radius and self.grid[row, col] in [2, 5]:
                    count += 1
        
        return count
    
    def _execute_rl_drone_action(self, action: list, drone_info: Dict) -> Dict[str, Any]:
        """Execute RL drone action for specific drone and return action info."""
        move_direction, action_type = action
        rl_data = drone_info['rl_data']
        
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
            'position': tuple(rl_data['position']),
            'drone_id': drone_info['id']
        }
        
        # Execute movement
        if move_direction in movement_directions:
            dr, dc = movement_directions[move_direction]
            new_row = rl_data['position'][0] + dr
            new_col = rl_data['position'][1] + dc
            
            if (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
                rl_data['position'] = [new_row, new_col]
                drone_info['position'] = [new_row, new_col]  # Update drone_info position too
                rl_data['energy'] -= 1  # Reduced energy consumption
                action_info['moved'] = True
            else:
                action_info['moved'] = False
        
        # Execute special action
        if action_type == 1:  # Fire suppression
            suppression_result = self._execute_rl_fire_suppression(drone_info)
            action_info.update(suppression_result)
        elif action_type == 2:  # Return to base
            base_result = self._execute_rl_return_to_base(drone_info)
            action_info.update(base_result)
        
        # Ensure resources don't go negative
        rl_data['water'] = max(0, rl_data['water'])
        rl_data['energy'] = max(0, rl_data['energy'])
        
        return action_info
    
    def _execute_rl_fire_suppression(self, drone_info: Dict) -> Dict[str, Any]:
        """Execute fire suppression for specific RL drone."""
        rl_data = drone_info['rl_data']
        
        if rl_data['water'] <= 0:
            return {'suppression_result': 'no_water'}
        
        row, col = rl_data['position']
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
                    
                    # Resource costs - Reduced consumption
                    water_cost = 2 if cell_type == 5 else 1  # Much lower water cost
                    rl_data['water'] -= water_cost
                    rl_data['energy'] -= 1  # Reduced energy cost
                    
                    fires_suppressed += 1
                    rl_data['fires_extinguished'] += 1
                    self.total_fires_extinguished += 1
        
        return {
            'suppression_result': 'success' if fires_suppressed > 0 else 'no_fire_in_range',
            'fires_suppressed': fires_suppressed
        }
    
    def _execute_rl_return_to_base(self, drone_info: Dict) -> Dict[str, Any]:
        """Execute return to base for specific RL drone."""
        rl_data = drone_info['rl_data']
        base_position = [0, 0]
        distance = np.sqrt((rl_data['position'][0] - base_position[0])**2 + 
                          (rl_data['position'][1] - base_position[1])**2)
        
        if distance <= 1.5:  # At base
            rl_data['water'] = rl_data['max_water']
            rl_data['energy'] = rl_data['max_energy']
            return {'base_action': 'refueled'}
        else:
            # Move towards base
            direction = [
                base_position[0] - rl_data['position'][0],
                base_position[1] - rl_data['position'][1]
            ]
            
            if abs(direction[0]) > abs(direction[1]):
                move = [1 if direction[0] > 0 else -1, 0]
            else:
                move = [0, 1 if direction[1] > 0 else -1]
            
            new_pos = [rl_data['position'][0] + move[0], 
                      rl_data['position'][1] + move[1]]
            
            if (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
                rl_data['position'] = new_pos
                drone_info['position'] = new_pos
                rl_data['energy'] -= 1
            
            return {'base_action': 'moving_to_base', 'distance': distance}
    
    def _get_rl_drone_status(self, drone_info: Dict) -> Dict[str, Any]:
        """Get RL drone status for display."""
        rl_data = drone_info['rl_data']
        return {
            'position': tuple(rl_data['position']),
            'water_level': rl_data['water'],
            'water_percentage': (rl_data['water'] / rl_data['max_water']) * 100,
            'energy': rl_data['energy'],
            'energy_percentage': (rl_data['energy'] / rl_data['max_energy']) * 100,
            'fires_extinguished': rl_data['fires_extinguished'],
            'known_fires': self._count_fires_in_radius(6, rl_data['position']),
            'target_fire': None,  # RL drone doesn't expose this directly
            'steps_taken': rl_data['step_count']
        }
    
    def _process_simple_drone_action(self, drone_action: Dict, drone_info: Dict):
        """Process action from simple drone."""
        # Get the actual drone object
        drone = drone_info['drone']
        
        # Sync our tracking position with the drone's actual position
        if drone and hasattr(drone, 'position'):
            drone_info['position'] = list(drone.position)
        
        # Process fire suppression
        if drone_action.get('action') == 'fire_suppressed':
            pos = drone_action.get('position')
            if pos and self._is_valid_position(pos):
                if self.grid[pos[0], pos[1]] in [self.FIRE, self.FIRE_INTENSE]:
                    self.grid[pos[0], pos[1]] = self.BURNED
                    self.fire_age[pos[0], pos[1]] = 0
                    self.total_fires_extinguished += 1
    
    
    def render(self, mode='human'):
        """Enhanced rendering for multi-drone system."""
        # Initialize pygame if needed
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size + 350, self.window_size + 100))
            pygame.display.set_caption("🔥 PyroGuard AI: Advanced Wildfire Simulation")
            self.clock = pygame.time.Clock()
            pygame.font.init()
        
        # Draw environment (but skip single drone rendering)
        self._draw_background()
        self.fire_animation_frame = (self.fire_animation_frame + 1) % 20
        self._draw_3d_terrain()
        self._draw_fire_effects()
        
        # Draw ALL drones instead of just one
        self._draw_all_drones()
        
        # Draw stats panel
        self._draw_stats_panel()
        
        pygame.display.flip()
        self.clock.tick(8)
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.spawn_random_fires(3)
                elif event.key == pygame.K_w:
                    self.change_wind()
                elif event.key == pygame.K_r:
                    self.reset()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_click(event.pos)
        
        return True
    
    def _draw_all_drones(self):
        """Draw all drones on the screen with different colors and IDs."""
        if not hasattr(self, 'drones') or not self.drones:
            return
        
        # Colors for different drone types and IDs - more distinct and bright
        drone_colors = [
            (50, 255, 50),    # Bright Green for first drone
            (50, 50, 255),    # Bright Blue for second drone  
            (255, 50, 255),   # Bright Magenta for third drone
            (255, 255, 50),   # Bright Yellow for fourth drone
            (50, 255, 255),   # Bright Cyan for fifth drone
            (255, 100, 50),   # Bright Orange for sixth drone
        ]
        
# Debug removed for cleaner output
        
        for i, drone_info in enumerate(self.drones):
            position = drone_info['position']
            drone_id = drone_info['id']
            drone_type = drone_info['type']
            
            self._draw_single_drone(position, drone_id, drone_type, drone_colors[i % len(drone_colors)])
    
    def _draw_single_drone(self, position: List[int], drone_id: int, drone_type: str, base_color: tuple):
        """Draw a single drone at the specified position."""
        if not hasattr(self, 'screen') or self.screen is None:
            return
        
        row, col = position
        
        # Calculate screen position - ensure it's within bounds
        x = col * self.cell_size + 20
        y = row * self.cell_size + 20
        
        # Simplified elevation for better visibility
        height_offset = 20  # Fixed height offset
        
        screen_x = x + self.cell_size // 2
        screen_y = y + self.cell_size // 2 - height_offset
        
        # Make drones larger and more visible
        drone_size = 16  # Increased from 12
        state_color = base_color
        
# Debug removed for cleaner output
        
        # For RL drones, modify color based on status
        if drone_type == 'rl':
            drone_data = next((d for d in self.drones if d['id'] == drone_id), None)
            if drone_data and 'rl_data' in drone_data:
                rl_data = drone_data['rl_data']
                if rl_data['water'] < 20:
                    state_color = (255, 255, 0)  # Yellow when low on water
                elif rl_data['energy'] < 30:
                    state_color = (255, 150, 0)  # Orange when low energy
            drone_size = 14  # RL drones slightly larger
        
        # Draw drone shadow
        shadow_pos = (screen_x + 2, screen_y + 15)
        pygame.draw.circle(self.screen, (30, 30, 30), shadow_pos, drone_size // 2)
        
        # Draw main drone body
        pygame.draw.circle(self.screen, (60, 60, 60), (screen_x, screen_y + 2), drone_size)
        pygame.draw.circle(self.screen, state_color, (screen_x, screen_y), drone_size)
        pygame.draw.circle(self.screen, (255, 255, 255), (screen_x - 3, screen_y - 3), drone_size // 3)
        
        # Draw type indicator
        if drone_type == 'rl':
            pygame.draw.circle(self.screen, (0, 255, 255), (screen_x, screen_y), drone_size // 4)
        else:
            pygame.draw.circle(self.screen, (255, 255, 255), (screen_x, screen_y), drone_size // 4)
        
        # Draw detection radius circle
        detection_radius = 5  # Default detection radius for simple drones
        if drone_type == 'rl':
            detection_radius = 6  # RL drones have slightly larger detection
        
        radius_pixels = detection_radius * self.cell_size
        radius_color = (*state_color[:3], 60)  # Semi-transparent version of drone color
        
        # Create surface for transparent circle
        radius_surface = pygame.Surface((radius_pixels * 2, radius_pixels * 2), pygame.SRCALPHA)
        pygame.draw.circle(radius_surface, radius_color, (radius_pixels, radius_pixels), radius_pixels, 2)
        self.screen.blit(radius_surface, (screen_x - radius_pixels, screen_y - radius_pixels))
        
        # Draw drone ID
        if hasattr(self, 'screen'):
            font = pygame.font.Font(None, 16)
            id_text = font.render(f"{drone_id}", True, (255, 255, 255))
            self.screen.blit(id_text, (screen_x - 4, screen_y + drone_size + 5))


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='PyroGuard AI Demo with Multi-Drone System')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained RL model')
    parser.add_argument('--grid-size', type=int, default=25,
                       help='Grid size (default: 25)')
    parser.add_argument('--num-drones', type=int, default=3,
                       help='Number of drones to deploy (default: 3)')
    parser.add_argument('--drone-types', type=str, nargs='+', default=None,
                       help='List of drone types: simple, rl (e.g., --drone-types rl simple simple)')
    parser.add_argument('--all-simple', action='store_true',
                       help='Use all simple drones')
    parser.add_argument('--all-rl', action='store_true',
                       help='Use all RL drones (requires model)')
    
    args = parser.parse_args()
    
    # Determine drone configuration
    num_drones = max(1, min(6, args.num_drones))  # Limit to 1-6 drones
    drone_types = None
    
    # Handle special flags
    if args.all_simple:
        drone_types = ["simple"] * num_drones
    elif args.all_rl:
        drone_types = ["rl"] * num_drones
    elif args.drone_types:
        drone_types = args.drone_types[:num_drones]
        # Pad with simple drones if not enough types specified
        while len(drone_types) < num_drones:
            drone_types.append("simple")
    
    # Check for RL model if needed
    if not args.model_path and (not drone_types or "rl" in drone_types):
        print("🔍 Looking for trained RL models...")
        
        # Look for available models
        if os.path.exists('models'):
            model_files = [f for f in os.listdir('models') if f.endswith('.pth')]
            if model_files:
                # Use the most recent model
                model_files.sort(reverse=True)
                args.model_path = os.path.join('models', model_files[0])
                print(f"📂 Found RL model: {args.model_path}")
            else:
                print("❌ No trained RL models found")
                if drone_types and "rl" in drone_types:
                    print("🔄 Converting RL drones to simple drones...")
                    drone_types = ["simple" if dt == "rl" else dt for dt in drone_types]
        else:
            print("❌ Models directory not found")
            if drone_types and "rl" in drone_types:
                print("🔄 Converting RL drones to simple drones...")
                drone_types = ["simple" if dt == "rl" else dt for dt in drone_types]
    
    
    # Display configuration
    drone_type_summary = ", ".join(drone_types) if drone_types else "auto (1 RL + others simple)"
    print(f"🔥 PyroGuard AI Demo - Multi-Drone System")
    print("=" * 60)
    print(f"🚁 Drones: {num_drones} ({drone_type_summary})")
    
    # Create multi-drone environment with slower fire spread
    env = MultiDroneWildfireEnvironment(
        grid_size=args.grid_size,
        fire_spread_prob=0.08,  # Reduced from 0.25 for slower spread
        initial_tree_density=0.75,
        wind_strength=0.08,     # Reduced wind for slower spread
        fire_persistence=8,     # Longer lasting fires
        new_fire_rate=0.015,    # Slower new fire spawning
        num_drones=num_drones,
        drone_types=drone_types,
        rl_model_path=args.model_path
    )
    
    # Initialize and run
    state = env.reset()
    print(f"🌲 Environment initialized with {num_drones} drones")
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
                fires_active = state.get('fires_active', 0)
                total_fires_extinguished = state.get('total_fires_extinguished', 0)
                num_active_drones = state.get('num_active_drones', 0)
                
                print(f"Step {step}: {fires_active} fires active, "
                      f"{total_fires_extinguished} extinguished by {num_active_drones} drones")
    
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    finally:
        env.close()
        print("🔥 Thanks for trying PyroGuard AI!")


if __name__ == "__main__":
    main()