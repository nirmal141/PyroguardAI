"""
Multi-agent firefighting environment with drones.
Extends the basic fire simulation to include multiple drone agents.
"""

from __future__ import annotations
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from typing import List, Dict, Tuple, Optional
import os
import time

from .fire_dynamics import TREE, BURNING, BURNED, DRONE_EXTINGUISHED, COMPLETELY_BURNED, ignite_random, step_fire, count_states
from .drone_agent import DroneAgent, DroneAction

class MultiAgentFireEnv(gym.Env):
    """
    Multi-agent firefighting environment with drones.
    
    Features:
    - Multiple drone agents with battery, water, and temperature sensors
    - RL-based firefighting coordination
    - Trigger-based activation system
    - Real-time visualization with drone status
    """
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(self, 
                 width=16, 
                 height=16, 
                 p_spread=0.3, 
                 p_burnout=0.1, 
                 max_steps=1000, 
                 ignitions=(1,3),
                 num_drones=3,
                 drone_start_positions=None,
                 seed: int | None = None):
        super().__init__()
        
        # Environment parameters
        self.width = int(width)
        self.height = int(height)
        self.p_spread = float(p_spread)
        self.p_burnout = float(p_burnout)
        self.max_steps = int(max_steps)
        self.ignitions_range = tuple(ignitions)
        self.num_drones = int(num_drones)
        
        # Drone management
        self.drones: List[DroneAgent] = []
        self.drone_start_positions = drone_start_positions or []
        self.drone_firefighting_active = False  # Trigger state
        
        # Action space: each drone can take 11 actions (8 movement directions, stay, extinguish, scan)
        self.action_space = spaces.MultiDiscrete([11] * self.num_drones)
        
        # Observation space: each drone observes its state + local grid
        # Each drone gets: [x, y, battery, water, temperature, fire_detected] + local_grid (3x3)
        drone_obs_size = 6 + 9  # 6 state vars + 3x3 local grid
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.num_drones, drone_obs_size), dtype=np.float32
        )
        
        # RNGs
        self.py_rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        
        # Environment state
        self.grid = None
        self.steps = 0
        self.screen = None
        self.cell_size = 50  # Bigger cells for better visibility
        
        # Performance tracking
        self.total_fires_extinguished = 0
        self.episode_rewards = []

    def _initialize_drones(self):
        """Initialize drone agents at specified or random positions."""
        self.drones = []
        
        for i in range(self.num_drones):
            if i < len(self.drone_start_positions):
                start_x, start_y = self.drone_start_positions[i]
            else:
                # Random position away from edges
                start_x = self.py_rng.randint(1, self.width - 2)
                start_y = self.py_rng.randint(1, self.height - 2)
            
            drone = DroneAgent(
                drone_id=i,
                start_x=start_x,
                start_y=start_y,
                max_battery=1.0,
                max_water=1.0,
                battery_drain_rate=0.005,  # Slower drain for longer missions
                water_usage_rate=0.15,
                temperature_threshold=0.7
            )
            self.drones.append(drone)

    def reset(self, seed: int | None = None, options=None):
        """Reset the environment to initial state."""
        if seed is not None:
            self.py_rng = random.Random(seed)
            self.np_rng = np.random.default_rng(seed)

        self.steps = 0
        self.drone_firefighting_active = False
        self.total_fires_extinguished = 0
        self.episode_rewards = []
        
        # Initialize fire grid
        self.grid = np.full((self.height, self.width), TREE, dtype=np.int8)
        
        # Ignite random fires
        k_min, k_max = self.ignitions_range
        k = self.py_rng.randint(k_min, k_max)
        ignite_random(self.grid, k=k, rng=self.np_rng)
        
        # Initialize drones
        self._initialize_drones()
        
        # Update drone temperature readings
        for drone in self.drones:
            drone.update_temperature_reading(self.grid, self.width, self.height)
        
        return self._get_observations(), {}

    def _get_observations(self) -> np.ndarray:
        """Get observations for all drones."""
        observations = np.zeros((self.num_drones, 6 + 9), dtype=np.float32)
        
        for i, drone in enumerate(self.drones):
            # Get drone's own state
            drone_obs = drone.get_observation(self.grid, self.width, self.height)
            observations[i, :6] = drone_obs
            
            # Get local 3x3 grid around drone
            local_grid = self._get_local_grid(drone.state.x, drone.state.y)
            observations[i, 6:] = local_grid.flatten()
        
        return observations

    def _get_local_grid(self, x: int, y: int, size: int = 3) -> np.ndarray:
        """Get local grid around position (x, y)."""
        local_grid = np.zeros((size, size), dtype=np.float32)
        half_size = size // 2
        
        for dy in range(size):
            for dx in range(size):
                nx, ny = x + dx - half_size, y + dy - half_size
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    # Normalize cell states: TREE=0, BURNING=1, BURNED=0.3, DRONE_EXTINGUISHED=0.5, COMPLETELY_BURNED=0.8
                    cell_state = self.grid[ny, nx]
                    if cell_state == BURNING:  # Burning
                        local_grid[dy, dx] = 1.0
                    elif cell_state == DRONE_EXTINGUISHED:  # Drone extinguished
                        local_grid[dy, dx] = 0.5
                    elif cell_state == BURNED:  # Natural burnt
                        local_grid[dy, dx] = 0.3
                    elif cell_state == COMPLETELY_BURNED:  # Completely burnt
                        local_grid[dy, dx] = 0.8
                    else:  # Tree
                        local_grid[dy, dx] = 0.0
                else:
                    local_grid[dy, dx] = -1.0  # Out of bounds
        
        return local_grid

    def step(self, actions: np.ndarray):
        """Execute actions for all drones and update environment."""
        self.steps += 1
        
        # Execute drone actions if firefighting is active
        drone_rewards = np.zeros(self.num_drones)
        if self.drone_firefighting_active:
            for i, (drone, action) in enumerate(zip(self.drones, actions)):
                if drone.state.is_active:
                    action_enum = DroneAction(action)
                    result = drone.execute_action(action_enum, self.grid, self.width, self.height)
                    drone_rewards[i] = result["reward"]
                    
                    # Handle fire extinguishing (only if drone actually extinguished a fire)
                    if result.get("message", "").startswith("Fire extinguished"):
                        # Change the cell from BURNING to DRONE_EXTINGUISHED (peach color)
                        if (0 <= drone.state.x < self.width and 
                            0 <= drone.state.y < self.height and 
                            self.grid[drone.state.y, drone.state.x] == BURNING):
                            self.grid[drone.state.y, drone.state.x] = DRONE_EXTINGUISHED
                            self.total_fires_extinguished += 1
        
        # Update fire dynamics
        old_burning = np.sum(self.grid == BURNING)
        self.grid = step_fire(self.grid, self.p_spread, self.p_burnout, rng=self.np_rng)
        new_burning = np.sum(self.grid == BURNING)
        
        # Calculate rewards
        fire_reduction_reward = (old_burning - new_burning) * 5.0  # Reward for fire reduction
        fire_penalty = -float(new_burning) * 0.1  # Small penalty for active fires
        
        # Battery and water management rewards
        battery_penalty = 0.0
        water_penalty = 0.0
        for drone in self.drones:
            if drone.state.battery < 0.2:  # Low battery warning
                battery_penalty -= 1.0
            if drone.state.water_capacity < 0.1:  # Low water warning
                water_penalty -= 0.5
        
        total_reward = fire_reduction_reward + fire_penalty + battery_penalty + water_penalty
        self.episode_rewards.append(total_reward)
        
        # Check termination conditions
        burning_cells = np.sum(self.grid == BURNING)
        terminated = burning_cells == 0  # All fires extinguished
        truncated = self.steps >= self.max_steps
        
        # Update drone temperature readings
        for drone in self.drones:
            drone.update_temperature_reading(self.grid, self.width, self.height)
        
        info = {
            "steps": self.steps,
            "burning": int(burning_cells),
            "drones_active": sum(1 for d in self.drones if d.state.is_active),
            "fires_extinguished": self.total_fires_extinguished,
            "drone_rewards": drone_rewards.tolist(),
            "firefighting_active": self.drone_firefighting_active
        }
        
        return self._get_observations(), total_reward, terminated, truncated, info

    def activate_drone_firefighting(self):
        """Activate drone firefighting mode (trigger button pressed)."""
        self.drone_firefighting_active = True
        print("🚁 Drone firefighting activated! Drones are now operational.")

    def deactivate_drone_firefighting(self):
        """Deactivate drone firefighting mode."""
        self.drone_firefighting_active = False
        print("🚁 Drone firefighting deactivated.")

    def recharge_drones(self):
        """Recharge all drones at base station."""
        for drone in self.drones:
            drone.recharge_battery(0.3)
            drone.refill_water(0.4)
        print("🔋 All drones recharged and refilled!")

    def render(self, mode="human"):
        """Render the environment with drones and fire."""
        if mode == "human":
            if not hasattr(self, "screen") or self.screen is None:
                pygame.init()
                # Make window with legend on the right side
                grid_width = self.width * self.cell_size
                grid_height = self.height * self.cell_size
                legend_width = 300  # Space for legend on the right
                window_width = grid_width + legend_width
                window_height = grid_height + 50  # Small space for bottom status
                self.screen = pygame.display.set_mode((window_width, window_height))
        
        if mode == "ansi":
            # ASCII representation
            symbols = {TREE: "🌲", BURNING: "🔥", BURNED: "🟤", DRONE_EXTINGUISHED: "🍑", COMPLETELY_BURNED: "⬛"}
            grid_str = "\n".join(
                "".join(symbols[int(self.grid[y, x])] for x in range(self.width)) 
                for y in range(self.height)
            )
            
            # Add drone positions
            drone_str = "\nDrones: "
            for drone in self.drones:
                if drone.state.is_active:
                    drone_str += f"🚁{drone.drone_id}({drone.state.x},{drone.state.y}) "
            
            return grid_str + drone_str

        if mode == "human":
            # Clear screen
            self.screen.fill((255, 255, 255))
            
            # Draw grid with correct color scheme
            colors = {TREE: (34, 139, 34), BURNING: (255, 0, 0), BURNED: (139, 69, 19), DRONE_EXTINGUISHED: (255, 218, 185), COMPLETELY_BURNED: (0, 0, 0)}  # Green, Red, Brown, Peach, Black
            symbols = {TREE: "🌲", BURNING: "🔥", BURNED: "🟤", DRONE_EXTINGUISHED: "🍑", COMPLETELY_BURNED: "⬛"}  # Tree, Fire, Natural Burnt, Drone Extinguished, Completely Burnt
            
            font_size = min(self.cell_size - 4, 24)
            font = pygame.font.Font(None, font_size)
            
            # Draw the grid
            for y in range(self.height):
                for x in range(self.width):
                    cell_state = int(self.grid[y, x])
                    
                    # Draw cell background
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                     self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, colors[cell_state], rect)
                    pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
                    
                    # Draw cell symbol
                    symbol = symbols[cell_state]
                    text_surface = font.render(symbol, True, (0, 0, 0))
                    text_rect = text_surface.get_rect()
                    text_rect.center = (x * self.cell_size + self.cell_size // 2,
                                      y * self.cell_size + self.cell_size // 2)
                    self.screen.blit(text_surface, text_rect)
            
            # Draw drones
            drone_font = pygame.font.Font(None, 20)
            for drone in self.drones:
                if drone.state.is_active:
                    # Draw drone
                    drone_x = drone.state.x * self.cell_size + self.cell_size // 2
                    drone_y = drone.state.y * self.cell_size + self.cell_size // 2
                    
                    # Drone color based on battery level
                    if drone.state.battery > 0.5:
                        drone_color = (0, 255, 0)  # Green
                    elif drone.state.battery > 0.2:
                        drone_color = (255, 255, 0)  # Yellow
                    else:
                        drone_color = (255, 0, 0)  # Red
                    
                    pygame.draw.circle(self.screen, drone_color, (drone_x, drone_y), 8)
                    pygame.draw.circle(self.screen, (0, 0, 0), (drone_x, drone_y), 8, 2)
                    
                    # Drone ID
                    id_text = drone_font.render(str(drone.drone_id), True, (0, 0, 0))
                    id_rect = id_text.get_rect()
                    id_rect.center = (drone_x, drone_y)
                    self.screen.blit(id_text, id_rect)
            
            # Draw legend on the right side
            legend_x = self.width * self.cell_size + 10
            legend_y = 20
            legend_font = pygame.font.Font(None, 24)
            small_font = pygame.font.Font(None, 20)
            
            # Legend title
            legend_title = legend_font.render("LEGEND", True, (0, 0, 0))
            self.screen.blit(legend_title, (legend_x, legend_y))
            
            # Legend items
            legend_items = [
                ("🌲", "Green", "Fresh Trees"),
                ("🔥", "Red", "Burning Fire"),
                ("🟤", "Brown", "Natural Burnt"),
                ("🍑", "Peach", "Drone Extinguished"),
                ("⬛", "Black", "Completely Burnt")
            ]
            
            for i, (symbol, color_name, description) in enumerate(legend_items):
                y_pos = legend_y + 40 + i * 30
                # Draw colored square
                color_rect = pygame.Rect(legend_x, y_pos, 20, 20)
                if i == 0:  # Green
                    pygame.draw.rect(self.screen, (34, 139, 34), color_rect)
                elif i == 1:  # Red
                    pygame.draw.rect(self.screen, (255, 0, 0), color_rect)
                elif i == 2:  # Brown
                    pygame.draw.rect(self.screen, (139, 69, 19), color_rect)
                elif i == 3:  # Peach
                    pygame.draw.rect(self.screen, (255, 218, 185), color_rect)
                elif i == 4:  # Black
                    pygame.draw.rect(self.screen, (0, 0, 0), color_rect)
                pygame.draw.rect(self.screen, (0, 0, 0), color_rect, 1)
                
                # Draw text
                legend_text = small_font.render(f"{symbol} {color_name}: {description}", True, (0, 0, 0))
                self.screen.blit(legend_text, (legend_x + 25, y_pos + 2))
            
            # Status information at bottom
            status_y = self.height * self.cell_size + 10
            status_font = pygame.font.Font(None, 20)
            
            # Compact status bar
            burning = np.sum(self.grid == BURNING)
            active_drones = sum(1 for d in self.drones if d.state.is_active)
            mode_text = "ACTIVE" if self.drone_firefighting_active else "INACTIVE"
            
            status_text = f"🔥 Fires: {burning} | 🚁 Drones: {active_drones}/{self.num_drones} | Firefighting: {mode_text} | ✅ Extinguished: {self.total_fires_extinguished}"
            status_display = status_font.render(status_text, True, (0, 0, 0))
            self.screen.blit(status_display, (10, status_y))
            
            # Controls reminder
            controls_text = small_font.render("Controls: SPACE=Pause, T=Activate, R=Recharge, ESC=Exit", True, (100, 100, 100))
            self.screen.blit(controls_text, (10, status_y + 25))
            
            pygame.display.flip()

    def close(self):
        """Clean up resources."""
        if self.screen:
            pygame.quit()
