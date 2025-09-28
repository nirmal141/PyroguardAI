
import numpy as np
import random
import math
from typing import Tuple, List


class FirefighterDrone:
    
    def __init__(self, start_pos: Tuple[int, int], grid_size: int):
      
        self.grid_size = grid_size
        self.position = list(start_pos)  # [row, col]
        
        # Drone capabilities
        self.detection_radius = 5  # Can detect fires within 5 cells
        self.suppression_range = 1  # Can suppress fires within 1 cell
        self.movement_speed = 2    # Moves 2 cells per step for faster response
        
        # Water and energy - Increased for longer missions
        self.max_water = 100  # 5x more water capacity
        self.water_level = self.max_water
        self.max_energy = 500  # 5x more energy capacity
        self.energy = self.max_energy
        
        # Mission state
        self.target_fire = None  # Current fire target (row, col)
        self.known_fires = []    # List of detected fires
        self.patrol_target = None  # Current patrol destination
        
        # Performance tracking
        self.fires_extinguished = 0
        self.steps_taken = 0
        
        # Visual trail for rendering
        self.trail = []
        self.max_trail_length = 8
    
    def update(self, environment_grid: np.ndarray) -> dict:
        
        self.steps_taken += 1
        
        # Consume energy - Reduced consumption for longer missions
        self.energy = max(0, self.energy - 0.5)
        
        # Update trail for visualization
        self.trail.append(tuple(self.position))
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)
        
        # Check if drone needs to return to base - Lower thresholds for longer missions
        if self.energy < 50 or self.water_level < 5:
            return self._return_to_base()
        
        # Scan for fires in detection radius
        detected_fires = self._scan_for_fires(environment_grid)
        self._update_known_fires(detected_fires, environment_grid)
        
        # Try to suppress nearby fires first
        suppression_result = self._try_suppress_fire(environment_grid)
        if suppression_result['action'] == 'fire_suppressed':
            return suppression_result
        
        # Move towards target fire or patrol
        movement_result = self._move_towards_target(environment_grid)
        
        return movement_result
    
    def _scan_for_fires(self, environment_grid: np.ndarray) -> List[Tuple[int, int]]:
        fires_detected = []
        row, col = self.position
        
        for dr in range(-self.detection_radius, self.detection_radius + 1):
            for dc in range(-self.detection_radius, self.detection_radius + 1):
                scan_row, scan_col = row + dr, col + dc
                
                # Check if position is valid
                if not self._is_valid_position((scan_row, scan_col)):
                    continue
                
                # Check distance
                distance = math.sqrt(dr*dr + dc*dc)
                if distance > self.detection_radius:
                    continue
                
                # Check for fire
                cell_type = environment_grid[scan_row, scan_col]
                if cell_type in [2, 5]:  # FIRE or FIRE_INTENSE
                    fires_detected.append((scan_row, scan_col))
        
        return fires_detected
    
    def _update_known_fires(self, detected_fires: List[Tuple[int, int]], 
                           environment_grid: np.ndarray):
        
        # Add newly detected fires
        for fire_pos in detected_fires:
            if fire_pos not in self.known_fires:
                self.known_fires.append(fire_pos)
        
        # Remove fires that are no longer burning
        self.known_fires = [
            fire_pos for fire_pos in self.known_fires
            if (self._is_valid_position(fire_pos) and 
                environment_grid[fire_pos[0], fire_pos[1]] in [2, 5])
        ]
        
        # Update target if current target is no longer valid
        if (self.target_fire and 
            (not self._is_valid_position(self.target_fire) or
             environment_grid[self.target_fire[0], self.target_fire[1]] not in [2, 5])):
            self.target_fire = None
    
    def _try_suppress_fire(self, environment_grid: np.ndarray) -> dict:
       
        if self.water_level == 0:
            return {'action': 'no_water', 'result': 'need_refill'}
        
        row, col = self.position
        
        # Check all positions within suppression range
        for dr in range(-self.suppression_range, self.suppression_range + 1):
            for dc in range(-self.suppression_range, self.suppression_range + 1):
                target_row, target_col = row + dr, col + dc
                
                if not self._is_valid_position((target_row, target_col)):
                    continue
                
                # Check if there's a fire here
                cell_type = environment_grid[target_row, target_col]
                if cell_type in [2, 5]:  # FIRE or FIRE_INTENSE
                    # Suppress the fire - Reduced water consumption
                    self.water_level = max(0, self.water_level - 1)
                    self.fires_extinguished += 1
                    
                    # Remove from known fires
                    fire_pos = (target_row, target_col)
                    if fire_pos in self.known_fires:
                        self.known_fires.remove(fire_pos)
                    
                    # Clear target if this was it
                    if self.target_fire == fire_pos:
                        self.target_fire = None
                    
                    return {
                        'action': 'fire_suppressed',
                        'position': fire_pos,
                        'water_remaining': self.water_level,
                        'fires_extinguished': self.fires_extinguished
                    }
        
        return {'action': 'no_fire_in_range', 'result': 'continue_mission'}
    
    def _move_towards_target(self, environment_grid: np.ndarray) -> dict:
       
        # Select target fire if we don't have one
        if not self.target_fire and self.known_fires:
            # Choose closest fire
            closest_fire = min(self.known_fires, 
                             key=lambda fire: self._distance_to(fire))
            self.target_fire = closest_fire
        
        # Move towards target fire
        if self.target_fire:
            moved = self._move_towards_position(self.target_fire)
            return {
                'action': 'move_to_fire',
                'target': self.target_fire,
                'moved': moved,
                'distance': self._distance_to(self.target_fire)
            }
        
        # No fires known, patrol randomly
        if not self.patrol_target or self._distance_to(self.patrol_target) < 1:
            self.patrol_target = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1)
            )
        
        moved = self._move_towards_position(self.patrol_target)
        return {
            'action': 'patrol',
            'target': self.patrol_target,
            'moved': moved
        }
    
    def _return_to_base(self) -> dict:
       
        base_position = (0, 0)  # Base at origin
        
        if self._distance_to(base_position) <= 1:
            # Refuel at base
            self.water_level = self.max_water
            self.energy = self.max_energy
            return {
                'action': 'refueled',
                'water_level': self.water_level,
                'energy_level': self.energy
            }
        else:
            # Move towards base
            moved = self._move_towards_position(base_position)
            return {
                'action': 'return_to_base',
                'moved': moved,
                'distance_to_base': self._distance_to(base_position)
            }
    
    def _move_towards_position(self, target_pos: Tuple[int, int]) -> bool:
       
        if not self._is_valid_position(target_pos):
            return False
        
        current_row, current_col = self.position
        target_row, target_col = target_pos
        
        # Calculate direction to move
        row_diff = target_row - current_row
        col_diff = target_col - current_col
        
        # If already at target
        if row_diff == 0 and col_diff == 0:
            return True
        
        # Move up to movement_speed cells per step
        moves_made = 0
        while moves_made < self.movement_speed and (row_diff != 0 or col_diff != 0):
            # Determine next move direction
            if abs(row_diff) > abs(col_diff):
                # Move vertically
                move_row = 1 if row_diff > 0 else -1
                move_col = 0
            elif abs(col_diff) > 0:
                # Move horizontally
                move_row = 0
                move_col = 1 if col_diff > 0 else -1
            else:
                break
            
            new_pos = (current_row + move_row, current_col + move_col)
            
            if self._is_valid_position(new_pos):
                self.position = list(new_pos)
                current_row, current_col = self.position
                row_diff = target_row - current_row
                col_diff = target_col - current_col
                moves_made += 1
            else:
                break
        
        return moves_made > 0
    
    def _distance_to(self, target_pos: Tuple[int, int]) -> float:
       
        return math.sqrt((self.position[0] - target_pos[0])**2 + 
                        (self.position[1] - target_pos[1])**2)
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
       
        row, col = pos
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size
    
    def get_status(self) -> dict:
       
        return {
            'position': tuple(self.position),
            'water_level': self.water_level,
            'water_percentage': (self.water_level / self.max_water) * 100,
            'energy': self.energy,
            'energy_percentage': (self.energy / self.max_energy) * 100,
            'fires_extinguished': self.fires_extinguished,
            'known_fires': len(self.known_fires),
            'target_fire': self.target_fire,
            'steps_taken': self.steps_taken
        }
