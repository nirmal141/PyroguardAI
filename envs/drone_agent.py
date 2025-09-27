"""
Drone agent for firefighting operations with battery, water capacity, and temperature sensors.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum

class DroneAction(Enum):
    """Available actions for a drone."""
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    MOVE_UP_LEFT = 4
    MOVE_UP_RIGHT = 5
    MOVE_DOWN_LEFT = 6
    MOVE_DOWN_RIGHT = 7
    STAY = 8
    EXTINGUISH = 9
    SCAN_TEMPERATURE = 10

@dataclass
class DroneState:
    """Current state of a drone."""
    x: int
    y: int
    battery: float  # 0.0 to 1.0
    water_capacity: float  # 0.0 to 1.0
    temperature_reading: float  # Current temperature at drone location
    is_active: bool  # Whether drone is operational
    last_action: Optional[DroneAction] = None

class DroneAgent:
    """
    Individual drone agent for firefighting operations.
    
    Key factors:
    - Battery: Decreases with movement and operations
    - Water capacity: Used for extinguishing fires
    - Temperature readings: Monitors fire intensity
    """
    
    def __init__(self, 
                 drone_id: int,
                 start_x: int, 
                 start_y: int,
                 max_battery: float = 1.0,
                 max_water: float = 1.0,
                 battery_drain_rate: float = 0.01,
                 water_usage_rate: float = 0.1,
                 temperature_threshold: float = 0.7):
        """
        Initialize a drone agent.
        
        Args:
            drone_id: Unique identifier for the drone
            start_x, start_y: Initial position
            max_battery: Maximum battery level (0.0 to 1.0)
            max_water: Maximum water capacity (0.0 to 1.0)
            battery_drain_rate: Battery consumed per action
            water_usage_rate: Water used per extinguishing action
            temperature_threshold: Temperature above which fire is detected
        """
        self.drone_id = drone_id
        self.state = DroneState(
            x=start_x,
            y=start_y,
            battery=max_battery,
            water_capacity=max_water,
            temperature_reading=0.0,
            is_active=True
        )
        
        # Configuration
        self.max_battery = max_battery
        self.max_water = max_water
        self.battery_drain_rate = battery_drain_rate
        self.water_usage_rate = water_usage_rate
        self.temperature_threshold = temperature_threshold
        
        # Performance tracking
        self.fires_extinguished = 0
        self.total_distance_traveled = 0
        
    def get_observation(self, grid: np.ndarray, grid_width: int, grid_height: int) -> np.ndarray:
        """
        Get the drone's observation of the environment.
        
        Returns:
            Array containing: [x, y, battery, water, temperature, fire_detected]
        """
        # Normalize position
        norm_x = self.state.x / (grid_width - 1) if grid_width > 1 else 0.0
        norm_y = self.state.y / (grid_height - 1) if grid_height > 1 else 0.0
        
        # Check if current cell is burning
        fire_detected = 1.0 if (0 <= self.state.x < grid_width and 
                               0 <= self.state.y < grid_height and 
                               grid[self.state.y, self.state.x] == 1) else 0.0
        
        return np.array([
            norm_x, norm_y,
            self.state.battery,
            self.state.water_capacity,
            self.state.temperature_reading,
            fire_detected
        ], dtype=np.float32)
    
    def update_temperature_reading(self, grid: np.ndarray, grid_width: int, grid_height: int):
        """Update temperature reading based on current position and surrounding fire."""
        if not (0 <= self.state.x < grid_width and 0 <= self.state.y < grid_height):
            self.state.temperature_reading = 0.0
            return
            
        # Base temperature from current cell
        current_cell = grid[self.state.y, self.state.x]
        if current_cell == 1:  # Burning
            base_temp = 1.0
        elif current_cell == 2:  # Burned
            base_temp = 0.3
        else:  # Tree
            base_temp = 0.0
            
        # Add temperature from nearby burning cells
        nearby_temp = 0.0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = self.state.x + dx, self.state.y + dy
                if (0 <= nx < grid_width and 0 <= ny < grid_height and 
                    grid[ny, nx] == 1):  # Nearby burning cell
                    distance = np.sqrt(dx*dx + dy*dy)
                    nearby_temp += 0.3 / distance  # Decrease with distance
                    
        self.state.temperature_reading = min(1.0, base_temp + nearby_temp)
    
    def can_perform_action(self, action: DroneAction) -> bool:
        """Check if the drone can perform the given action."""
        if not self.state.is_active or self.state.battery <= 0:
            return False
            
        if action == DroneAction.EXTINGUISH:
            return self.state.water_capacity > 0
            
        return True
    
    def execute_action(self, action: DroneAction, grid: np.ndarray, 
                      grid_width: int, grid_height: int) -> dict:
        """
        Execute an action and return the result.
        
        Returns:
            Dictionary with action results and rewards
        """
        if not self.can_perform_action(action):
            return {"success": False, "reward": -0.1, "message": "Cannot perform action"}
        
        result = {"success": True, "reward": 0.0, "message": ""}
        old_x, old_y = self.state.x, self.state.y
        
        # Execute movement actions (including diagonal)
        if action == DroneAction.MOVE_UP and self.state.y > 0:
            self.state.y -= 1
            self.total_distance_traveled += 1
        elif action == DroneAction.MOVE_DOWN and self.state.y < grid_height - 1:
            self.state.y += 1
            self.total_distance_traveled += 1
        elif action == DroneAction.MOVE_LEFT and self.state.x > 0:
            self.state.x -= 1
            self.total_distance_traveled += 1
        elif action == DroneAction.MOVE_RIGHT and self.state.x < grid_width - 1:
            self.state.x += 1
            self.total_distance_traveled += 1
        elif action == DroneAction.MOVE_UP_LEFT and self.state.y > 0 and self.state.x > 0:
            self.state.y -= 1
            self.state.x -= 1
            self.total_distance_traveled += 1
        elif action == DroneAction.MOVE_UP_RIGHT and self.state.y > 0 and self.state.x < grid_width - 1:
            self.state.y -= 1
            self.state.x += 1
            self.total_distance_traveled += 1
        elif action == DroneAction.MOVE_DOWN_LEFT and self.state.y < grid_height - 1 and self.state.x > 0:
            self.state.y += 1
            self.state.x -= 1
            self.total_distance_traveled += 1
        elif action == DroneAction.MOVE_DOWN_RIGHT and self.state.y < grid_height - 1 and self.state.x < grid_width - 1:
            self.state.y += 1
            self.state.x += 1
            self.total_distance_traveled += 1
        elif action == DroneAction.STAY:
            pass  # No movement
        elif action == DroneAction.EXTINGUISH:
            # Try to extinguish fire at current location
            if (0 <= self.state.x < grid_width and 
                0 <= self.state.y < grid_height and 
                grid[self.state.y, self.state.x] == 1):  # Burning cell
                # Successfully extinguish fire
                result["reward"] = 10.0  # High reward for extinguishing
                result["message"] = "Fire extinguished!"
                self.fires_extinguished += 1
                self.state.water_capacity = max(0.0, self.state.water_capacity - self.water_usage_rate)
            else:
                result["reward"] = -1.0  # Penalty for trying to extinguish non-burning cell
                result["message"] = "No fire to extinguish"
        elif action == DroneAction.SCAN_TEMPERATURE:
            # Update temperature reading
            self.update_temperature_reading(grid, grid_width, grid_height)
            result["reward"] = 0.1  # Small reward for gathering information
            result["message"] = f"Temperature: {self.state.temperature_reading:.2f}"
        
        # Drain battery for any action
        self.state.battery = max(0.0, self.state.battery - self.battery_drain_rate)
        
        # Check if drone is still operational
        if self.state.battery <= 0:
            self.state.is_active = False
            result["message"] += " Drone battery depleted!"
            result["reward"] -= 5.0  # Penalty for battery depletion
        
        # Update last action
        self.state.last_action = action
        
        # Update temperature reading after movement
        if action in [DroneAction.MOVE_UP, DroneAction.MOVE_DOWN, 
                     DroneAction.MOVE_LEFT, DroneAction.MOVE_RIGHT, DroneAction.STAY]:
            self.update_temperature_reading(grid, grid_width, grid_height)
        
        return result
    
    def recharge_battery(self, amount: float = 0.2):
        """Recharge drone battery (for base station operations)."""
        self.state.battery = min(self.max_battery, self.state.battery + amount)
        if self.state.battery > 0:
            self.state.is_active = True
    
    def refill_water(self, amount: float = 0.3):
        """Refill drone water tank (for base station operations)."""
        self.state.water_capacity = min(self.max_water, self.state.water_capacity + amount)
    
    def get_status(self) -> dict:
        """Get current drone status for monitoring."""
        return {
            "drone_id": self.drone_id,
            "position": (self.state.x, self.state.y),
            "battery": self.state.battery,
            "water": self.state.water_capacity,
            "temperature": self.state.temperature_reading,
            "is_active": self.state.is_active,
            "fires_extinguished": self.fires_extinguished,
            "distance_traveled": self.total_distance_traveled
        }
