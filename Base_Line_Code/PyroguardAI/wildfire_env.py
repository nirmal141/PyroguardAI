import numpy as np
import pygame
import random
import math
from typing import Tuple, List, Optional, Dict
import gymnasium as gym
from gymnasium import spaces
from simple_drone import FirefighterDrone

class WildfireEnvironment:
    """
    Basic wildfire environment with grid-based forest simulation and random fire spawning.
    Enhanced for hackathon demonstrations with longer-lasting fires and dramatic spread.
    """
    
    # Cell types - More realistic terrain
    EMPTY = 0          # Bare ground/dirt
    TREE = 1           # Healthy forest
    FIRE = 2           # Active fire
    BURNED = 3         # Burned area
    WATER = 4          # Water bodies (rivers, lakes)
    FIRE_INTENSE = 5   # Intense fire that burns longer
    DENSE_FOREST = 6   # Dense forest (burns faster, spreads more)
    SPARSE_FOREST = 7  # Sparse forest (burns slower)
    ROCK = 8           # Rocky terrain (firebreak)
    GRASS = 9          # Grassland (burns fast but less intense)
    
    # Enhanced colors for 3D-like visualization
    COLORS = {
        EMPTY: (139, 69, 19),      # Brown (dirt)
        TREE: (34, 139, 34),       # Forest Green
        FIRE: (255, 69, 0),        # Red Orange
        BURNED: (64, 64, 64),      # Dark Gray
        WATER: (0, 191, 255),      # Deep Sky Blue
        FIRE_INTENSE: (255, 0, 0), # Bright Red (intense fire)
        DENSE_FOREST: (0, 100, 0), # Dark Green (dense forest)
        SPARSE_FOREST: (144, 238, 144), # Light Green (sparse)
        ROCK: (105, 105, 105),     # Gray (rocky terrain)
        GRASS: (154, 205, 50)      # Yellow Green (grassland)
    }
    
    # Shadow colors for 3D effect
    SHADOW_COLORS = {
        EMPTY: (90, 45, 12),       # Darker brown
        TREE: (20, 80, 20),        # Darker green
        FIRE: (180, 40, 0),        # Darker red
        BURNED: (40, 40, 40),      # Darker gray
        WATER: (0, 120, 180),      # Darker blue
        FIRE_INTENSE: (180, 0, 0), # Darker red
        DENSE_FOREST: (0, 60, 0),  # Darker dense green
        SPARSE_FOREST: (100, 180, 100), # Darker light green
        ROCK: (70, 70, 70),        # Darker gray
        GRASS: (120, 160, 35)      # Darker yellow-green
    }
    
    # Highlight colors for 3D effect
    HIGHLIGHT_COLORS = {
        EMPTY: (180, 90, 25),      # Lighter brown
        TREE: (50, 180, 50),       # Lighter green
        FIRE: (255, 120, 50),      # Lighter red-orange
        BURNED: (90, 90, 90),      # Lighter gray
        WATER: (100, 220, 255),    # Lighter blue
        FIRE_INTENSE: (255, 100, 100), # Lighter red
        DENSE_FOREST: (30, 150, 30), # Lighter dense green
        SPARSE_FOREST: (180, 255, 180), # Lighter sparse green
        ROCK: (140, 140, 140),     # Lighter gray
        GRASS: (200, 255, 80)      # Lighter yellow-green
    }
    
    def __init__(self, grid_size: int = 30, fire_spread_prob: float = 0.15, 
                 initial_tree_density: float = 0.75, wind_strength: float = 0.15,
                 fire_persistence: int = 6, new_fire_rate: float = 0.01,
                 enable_drones: bool = True):
        """
        Initialize the wildfire environment.
        
        Args:
            grid_size: Size of the square grid
            fire_spread_prob: Base probability of fire spreading to adjacent cells
            initial_tree_density: Percentage of grid initially covered with trees
            wind_strength: Strength of wind effect on fire spread (0.0 to 0.5)
            fire_persistence: Number of steps a fire burns before extinguishing
            new_fire_rate: Probability of new fires spawning each step
            enable_drones: Whether to enable AI drone system
        """
        self.grid_size = grid_size
        self.fire_spread_prob = fire_spread_prob
        self.initial_tree_density = initial_tree_density
        self.wind_strength = wind_strength
        self.fire_persistence = fire_persistence
        self.new_fire_rate = new_fire_rate
        self.enable_drones = enable_drones
        
        # Wind direction (in degrees, 0 = North, 90 = East, 180 = South, 270 = West)
        self.wind_direction = random.uniform(0, 360)
        
        # Initialize grid and fire age tracking
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.fire_age = np.zeros((grid_size, grid_size), dtype=int)  # Track how long fires have been burning
        self.reset()
        
        # Pygame setup for 3D-like visualization
        self.cell_size = 22  # Slightly larger for better 3D effect
        self.window_size = grid_size * self.cell_size
        self.screen = None
        self.clock = None
        
        # 3D rendering parameters
        self.elevation_scale = 8  # Height variation for 3D effect
        self.shadow_offset = 2    # Shadow depth
        self.fire_animation_frame = 0  # For animated fire effects
        
        # Statistics
        self.step_count = 0
        self.total_trees_burned = 0
        self.total_fires_started = 0
        self.fires_extinguished = 0
        
        # Initialize drone system
        self.drone = None
        if self.enable_drones:
            self._initialize_drone()
        
    def reset(self):
        """Reset the environment to initial state with realistic terrain generation."""
        self.grid.fill(self.EMPTY)
        self.fire_age.fill(0)
        self.step_count = 0
        self.total_trees_burned = 0
        self.total_fires_started = 0
        self.fires_extinguished = 0
        
        # Generate realistic terrain features
        self._generate_realistic_terrain()
        
        # Reset drone system
        if self.enable_drones:
            self._initialize_drone()
        
        # Start with no fires - let user control when fires begin
        return self.get_state()
    
    def _generate_realistic_terrain(self):
        """Generate realistic terrain with forests, water bodies, rocks, and grasslands."""
        # Start with base terrain
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Create elevation-based terrain variation
                elevation = self._get_elevation(i, j)
                
                if elevation < 0.2:  # Low areas - water and wet areas
                    if random.random() < 0.3:
                        self.grid[i, j] = self.WATER
                    else:
                        self.grid[i, j] = self.GRASS
                elif elevation < 0.4:  # Mid-low areas - grassland and sparse forest
                    if random.random() < 0.6:
                        self.grid[i, j] = self.SPARSE_FOREST
                    else:
                        self.grid[i, j] = self.GRASS
                elif elevation < 0.7:  # Mid areas - main forest
                    if random.random() < 0.8:
                        self.grid[i, j] = self.TREE
                    else:
                        self.grid[i, j] = self.SPARSE_FOREST
                elif elevation < 0.9:  # High areas - dense forest
                    if random.random() < 0.7:
                        self.grid[i, j] = self.DENSE_FOREST
                    else:
                        self.grid[i, j] = self.TREE
                else:  # Very high areas - rocky terrain
                    self.grid[i, j] = self.ROCK
        
        # Add water features (rivers, lakes)
        self._add_water_features()
        
        # Add some clearings and firebreaks
        self._add_clearings()
    
    def _get_elevation(self, i: int, j: int) -> float:
        """Generate realistic elevation using simple noise-like function."""
        # Create elevation variation using distance from multiple points
        center_x, center_y = self.grid_size // 2, self.grid_size // 2
        
        # Distance from center (creates mountain-like elevation)
        dist_center = np.sqrt((i - center_x)**2 + (j - center_y)**2)
        base_elevation = 1.0 - (dist_center / (self.grid_size * 0.7))
        
        # Add some randomness and secondary peaks
        noise = random.random() * 0.3 - 0.15
        secondary_peak_1 = 0.2 * np.exp(-((i - self.grid_size*0.3)**2 + (j - self.grid_size*0.3)**2) / 50)
        secondary_peak_2 = 0.15 * np.exp(-((i - self.grid_size*0.7)**2 + (j - self.grid_size*0.8)**2) / 40)
        
        elevation = base_elevation + noise + secondary_peak_1 + secondary_peak_2
        return max(0, min(1, elevation))
    
    def _add_water_features(self):
        """Add rivers and lakes to the terrain."""
        # Add a river running through the map
        river_start_x = random.randint(0, self.grid_size - 1)
        river_y = 0
        
        for y in range(self.grid_size):
            # River meanders slightly
            river_x = river_start_x + int(3 * np.sin(y * 0.3)) + random.randint(-1, 1)
            river_x = max(1, min(self.grid_size - 2, river_x))
            
            # River width varies
            width = random.randint(1, 3)
            for w in range(-width//2, width//2 + 1):
                x = river_x + w
                if 0 <= x < self.grid_size:
                    self.grid[y, x] = self.WATER
        
        # Add a few small lakes
        num_lakes = random.randint(2, 4)
        for _ in range(num_lakes):
            lake_center_x = random.randint(3, self.grid_size - 4)
            lake_center_y = random.randint(3, self.grid_size - 4)
            lake_size = random.randint(2, 4)
            
            for dx in range(-lake_size, lake_size + 1):
                for dy in range(-lake_size, lake_size + 1):
                    if dx*dx + dy*dy <= lake_size*lake_size:
                        x, y = lake_center_x + dx, lake_center_y + dy
                        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                            self.grid[y, x] = self.WATER
    
    def _add_clearings(self):
        """Add clearings and firebreaks for realism."""
        # Add some clearings
        num_clearings = random.randint(3, 6)
        for _ in range(num_clearings):
            clearing_x = random.randint(2, self.grid_size - 3)
            clearing_y = random.randint(2, self.grid_size - 3)
            clearing_size = random.randint(2, 4)
            
            for dx in range(-clearing_size, clearing_size + 1):
                for dy in range(-clearing_size, clearing_size + 1):
                    x, y = clearing_x + dx, clearing_y + dy
                    if (0 <= x < self.grid_size and 0 <= y < self.grid_size and
                        dx*dx + dy*dy <= clearing_size*clearing_size):
                        if random.random() < 0.7:
                            self.grid[y, x] = self.GRASS
    
    def _initialize_drone(self):
        """Initialize a single firefighter drone."""
        start_position = (1, 1)  # Start at base
        self.drone = FirefighterDrone(start_position, self.grid_size)
    
    def _process_drone_action(self, drone_action: Dict):
        """Process drone action that affects the environment."""
        if drone_action.get('action') == 'fire_suppressed':
            # Drone successfully suppressed a fire
            pos = drone_action.get('position')
            if pos and self._is_valid_position(pos):
                if self.grid[pos[0], pos[1]] in [self.FIRE, self.FIRE_INTENSE]:
                    self.grid[pos[0], pos[1]] = self.BURNED
                    self.fire_age[pos[0], pos[1]] = 0
                    self.fires_extinguished += 1
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds."""
        row, col = pos
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size
    
    def spawn_random_fires(self, num_fires: int = 1):
        """Spawn random fires in flammable vegetation cells."""
        # Find all flammable positions
        flammable_mask = ((self.grid == self.TREE) | 
                         (self.grid == self.DENSE_FOREST) | 
                         (self.grid == self.SPARSE_FOREST) |
                         (self.grid == self.GRASS))
        flammable_positions = np.where(flammable_mask)
        
        if len(flammable_positions[0]) == 0:
            return  # No flammable vegetation to burn
        
        available_positions = list(zip(flammable_positions[0], flammable_positions[1]))
        
        for _ in range(min(num_fires, len(available_positions))):
            if available_positions:
                pos = random.choice(available_positions)
                self.grid[pos[0], pos[1]] = self.FIRE
                available_positions.remove(pos)
                self.total_fires_started += 1
    
    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get valid neighboring positions (4-connected)."""
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                neighbors.append((new_row, new_col))
        return neighbors
    
    def calculate_wind_effect(self, fire_pos: Tuple[int, int], target_pos: Tuple[int, int]) -> float:
        """Calculate wind effect on fire spread probability."""
        fire_row, fire_col = fire_pos
        target_row, target_col = target_pos
        
        # Calculate direction from fire to target
        direction_vector = (target_row - fire_row, target_col - fire_col)
        
        # Convert wind direction to vector (North is negative row direction)
        wind_rad = np.radians(self.wind_direction)
        wind_vector = (-np.cos(wind_rad), np.sin(wind_rad))
        
        # Calculate dot product to see if wind helps spread in this direction
        if direction_vector == (0, 0):
            return 0
        
        # Normalize direction vector
        direction_magnitude = np.sqrt(direction_vector[0]**2 + direction_vector[1]**2)
        normalized_direction = (direction_vector[0] / direction_magnitude, 
                              direction_vector[1] / direction_magnitude)
        
        # Dot product gives cosine of angle between wind and spread direction
        dot_product = (wind_vector[0] * normalized_direction[0] + 
                      wind_vector[1] * normalized_direction[1])
        
        # Wind effect ranges from -wind_strength to +wind_strength
        return dot_product * self.wind_strength
    
    def step(self):
        """Advance the simulation by one time step."""
        self.step_count += 1
        
        # Find all current fire positions (both regular and intense fires)
        fire_positions = np.where((self.grid == self.FIRE) | (self.grid == self.FIRE_INTENSE))
        current_fires = list(zip(fire_positions[0], fire_positions[1]))
        
        # Track new fires and fires to extinguish
        new_fires = []
        fires_to_extinguish = []
        
        # Process each fire
        for fire_row, fire_col in current_fires:
            # Age the fire
            self.fire_age[fire_row, fire_col] += 1
            
            # Check if fire should burn out
            if self.fire_age[fire_row, fire_col] >= self.fire_persistence:
                fires_to_extinguish.append((fire_row, fire_col))
                continue
            
            # Make fire more intense as it ages
            if (self.fire_age[fire_row, fire_col] >= 2 and 
                self.grid[fire_row, fire_col] == self.FIRE):
                self.grid[fire_row, fire_col] = self.FIRE_INTENSE
            
            # Try to spread to neighbors
            neighbors = self.get_neighbors(fire_row, fire_col)
            for neighbor_row, neighbor_col in neighbors:
                neighbor_cell = self.grid[neighbor_row, neighbor_col]
                
                # Check if neighbor is flammable
                if neighbor_cell in [self.TREE, self.DENSE_FOREST, self.SPARSE_FOREST, self.GRASS]:
                    # Calculate spread probability with wind effect
                    wind_effect = self.calculate_wind_effect((fire_row, fire_col), 
                                                           (neighbor_row, neighbor_col))
                    
                    # Different terrain types have different flammability
                    base_prob = self.fire_spread_prob
                    if neighbor_cell == self.DENSE_FOREST:
                        base_prob *= 1.8  # Dense forest burns much easier
                    elif neighbor_cell == self.SPARSE_FOREST:
                        base_prob *= 0.6  # Sparse forest burns slower
                    elif neighbor_cell == self.GRASS:
                        base_prob *= 1.3  # Grass burns fast but less intense
                    
                    # Intense fires spread more easily
                    if self.grid[fire_row, fire_col] == self.FIRE_INTENSE:
                        base_prob *= 1.5  # 50% more likely to spread
                    
                    spread_prob = base_prob + wind_effect
                    spread_prob = max(0, min(1, spread_prob))  # Clamp to [0, 1]
                    
                    if random.random() < spread_prob:
                        new_fires.append((neighbor_row, neighbor_col))
        
        # Extinguish old fires
        for row, col in fires_to_extinguish:
            self.grid[row, col] = self.BURNED
            self.fire_age[row, col] = 0
            self.total_trees_burned += 1
            self.fires_extinguished += 1
        
        # Start new fires
        for row, col in new_fires:
            cell_type = self.grid[row, col]
            # Double-check it's still flammable vegetation
            if cell_type in [self.TREE, self.DENSE_FOREST, self.SPARSE_FOREST, self.GRASS]:
                self.grid[row, col] = self.FIRE
                self.fire_age[row, col] = 0
        
        # Only spawn automatic fires if there are already fires burning (to keep action going)
        current_fire_count = len(current_fires)
        if current_fire_count > 0 and random.random() < (self.new_fire_rate * 0.3):  # Much less frequent
            self.spawn_random_fires(1)
        
        # Occasionally change wind direction and strength
        if random.random() < 0.15:  # 15% chance per step
            self.wind_direction = random.uniform(0, 360)
            # Occasionally make wind stronger for more dramatic spread
            if random.random() < 0.3:
                self.wind_strength = random.uniform(0.1, 0.3)
        
        # Update drone system
        drone_action = {}
        if self.enable_drones and self.drone:
            drone_action = self.drone.update(self.grid)
            # Process drone fire suppression actions
            self._process_drone_action(drone_action)
        
        state = self.get_state()
        if self.enable_drones:
            state['drone_action'] = drone_action
        
        return state
    
    def get_state(self) -> dict:
        """Get current state of the environment."""
        fire_count = np.sum((self.grid == self.FIRE) | (self.grid == self.FIRE_INTENSE))
        
        # Count all vegetation types
        tree_count = np.sum(self.grid == self.TREE)
        dense_forest_count = np.sum(self.grid == self.DENSE_FOREST)
        sparse_forest_count = np.sum(self.grid == self.SPARSE_FOREST)
        grass_count = np.sum(self.grid == self.GRASS)
        total_vegetation = tree_count + dense_forest_count + sparse_forest_count + grass_count
        
        burned_count = np.sum(self.grid == self.BURNED)
        intense_fire_count = np.sum(self.grid == self.FIRE_INTENSE)
        water_count = np.sum(self.grid == self.WATER)
        rock_count = np.sum(self.grid == self.ROCK)
        
        return {
            'grid': self.grid.copy(),
            'fire_age': self.fire_age.copy(),
            'step': self.step_count,
            'fires_active': fire_count,
            'fires_intense': intense_fire_count,
            'trees_remaining': tree_count,
            'total_vegetation': total_vegetation,
            'dense_forest': dense_forest_count,
            'sparse_forest': sparse_forest_count,
            'grassland': grass_count,
            'trees_burned': burned_count,
            'total_trees_burned': self.total_trees_burned,
            'total_fires_started': self.total_fires_started,
            'fires_extinguished': self.fires_extinguished,
            'water_bodies': water_count,
            'rocky_terrain': rock_count,
            'wind_direction': self.wind_direction,
            'wind_strength': self.wind_strength,
            # Never end automatically - let it run forever for demo control
            'is_done': False
        }
        
        # Add drone information if enabled
        if self.enable_drones and self.drone:
            state['drone_status'] = self.drone.get_status()
        
        return state
    
    def render(self, mode='human'):
        """Render the environment using enhanced 3D-like visualization."""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size + 350, self.window_size + 100))
            pygame.display.set_caption("🔥 PyroGuard AI: Advanced Wildfire Simulation")
            self.clock = pygame.time.Clock()
            # Load or create fonts for better text rendering
            pygame.font.init()
        
        # Enhanced background with gradient effect
        self._draw_background()
        
        # Update fire animation frame
        self.fire_animation_frame = (self.fire_animation_frame + 1) % 20
        
        # Draw 3D-like terrain grid
        self._draw_3d_terrain()
        
        # Draw animated fire effects
        self._draw_fire_effects()
        
        # Draw drone system if enabled
        if self.enable_drones and self.drone:
            self._draw_drone()
        
        # Draw enhanced statistics panel with modern styling
        self._draw_stats_panel()
        
        pygame.display.flip()
        self.clock.tick(8)  # Slightly faster for smoother animations
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Spacebar: Spawn new fires for dramatic effect
                    self.spawn_random_fires(random.randint(2, 4))
                    print("🔥 New fires spawned!")
                elif event.key == pygame.K_w:
                    # W key: Change wind direction randomly
                    self.wind_direction = random.uniform(0, 360)
                    self.wind_strength = random.uniform(0.1, 0.4)
                    print(f"💨 Wind changed! Direction: {self.wind_direction:.1f}°, Strength: {self.wind_strength:.2f}")
                elif event.key == pygame.K_r:
                    # R key: Reset environment
                    self.reset()
                    print("🌲 Environment reset! Fresh forest ready.")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Click to spawn fire at location
                    mouse_x, mouse_y = event.pos
                    if mouse_x < self.window_size and mouse_y < self.window_size:
                        grid_x = mouse_x // self.cell_size
                        grid_y = mouse_y // self.cell_size
                        if (0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size):
                            cell_type = self.grid[grid_y, grid_x]
                            if cell_type in [self.TREE, self.DENSE_FOREST, self.SPARSE_FOREST, self.GRASS]:
                                self.grid[grid_y, grid_x] = self.FIRE
                                self.fire_age[grid_y, grid_x] = 0
                                terrain_names = {
                                    self.TREE: "tree",
                                    self.DENSE_FOREST: "dense forest", 
                                    self.SPARSE_FOREST: "sparse forest",
                                    self.GRASS: "grassland"
                                }
                                print(f"🔥 Fire ignited in {terrain_names[cell_type]} at ({grid_x}, {grid_y})!")
                            elif cell_type == self.WATER:
                                print(f"💧 Can't ignite water at ({grid_x}, {grid_y})")
                            elif cell_type == self.ROCK:
                                print(f"🗿 Can't ignite rock at ({grid_x}, {grid_y})")
        
        return True
    
    
    def _draw_background(self):
        """Draw enhanced gradient background."""
        # Create a gradient from dark blue (sky) to dark green (ground)
        for y in range(self.window_size + 100):
            # Gradient from dark blue at top to dark green at bottom
            ratio = y / (self.window_size + 100)
            r = int(20 + ratio * 15)  # 20 -> 35
            g = int(30 + ratio * 20)  # 30 -> 50  
            b = int(50 - ratio * 30)  # 50 -> 20
            color = (r, g, b)
            pygame.draw.line(self.screen, color, (0, y), (self.window_size + 350, y))
    
    def _draw_3d_terrain(self):
        """Draw terrain with 3D-like effects using shadows and highlights."""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell_type = self.grid[i, j]
                
                # Calculate 3D position
                x = j * self.cell_size + 20  # Offset from edge
                y = i * self.cell_size + 20
                
                # Get elevation for this cell (reuse from terrain generation)
                elevation = self._get_elevation(i, j)
                height_offset = int(elevation * self.elevation_scale)
                
                # Draw shadow first (behind and below)
                shadow_rect = pygame.Rect(
                    x + self.shadow_offset, 
                    y + self.shadow_offset + height_offset, 
                    self.cell_size - self.shadow_offset, 
                    self.cell_size - self.shadow_offset
                )
                pygame.draw.rect(self.screen, self.SHADOW_COLORS[cell_type], shadow_rect)
                
                # Draw main cell
                main_rect = pygame.Rect(x, y - height_offset, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.COLORS[cell_type], main_rect)
                
                # Add highlight on top and left edges for 3D effect
                highlight_color = self.HIGHLIGHT_COLORS[cell_type]
                
                # Top highlight
                pygame.draw.line(self.screen, highlight_color, 
                               (x, y - height_offset), 
                               (x + self.cell_size, y - height_offset), 2)
                
                # Left highlight
                pygame.draw.line(self.screen, highlight_color,
                               (x, y - height_offset),
                               (x, y + self.cell_size - height_offset), 2)
                
                # Add special effects for certain terrain types
                self._add_terrain_details(x, y - height_offset, cell_type, elevation)
    
    def _add_terrain_details(self, x: int, y: int, cell_type: int, elevation: float):
        """Add special visual details for different terrain types."""
        if cell_type == self.WATER:
            # Add water ripple effects
            for ripple in range(3):
                ripple_alpha = 100 - ripple * 30
                ripple_color = (255, 255, 255, ripple_alpha)
                offset = (self.fire_animation_frame + ripple * 5) % 10
                if offset < 5:
                    pygame.draw.circle(self.screen, (150, 200, 255), 
                                     (x + self.cell_size//2, y + self.cell_size//2), 
                                     offset + 2, 1)
        
        elif cell_type == self.ROCK:
            # Add rocky texture with small dots
            for _ in range(3):
                dot_x = x + random.randint(2, self.cell_size - 2)
                dot_y = y + random.randint(2, self.cell_size - 2)
                pygame.draw.circle(self.screen, (80, 80, 80), (dot_x, dot_y), 1)
        
        elif cell_type in [self.TREE, self.DENSE_FOREST, self.SPARSE_FOREST]:
            # Add tree canopy effect with small circles
            canopy_color = self.HIGHLIGHT_COLORS[cell_type]
            for _ in range(2 if cell_type == self.SPARSE_FOREST else 4):
                canopy_x = x + random.randint(3, self.cell_size - 3)
                canopy_y = y + random.randint(3, self.cell_size - 3)
                pygame.draw.circle(self.screen, canopy_color, (canopy_x, canopy_y), 2)
    
    def _draw_fire_effects(self):
        """Draw animated fire effects with particles and glow."""
        fire_positions = np.where((self.grid == self.FIRE) | (self.grid == self.FIRE_INTENSE))
        
        for i, j in zip(fire_positions[0], fire_positions[1]):
            x = j * self.cell_size + 20
            y = i * self.cell_size + 20
            
            # Get elevation for positioning
            elevation = self._get_elevation(i, j)
            height_offset = int(elevation * self.elevation_scale)
            
            # Draw fire glow effect (larger background circle)
            glow_radius = 15 + int(3 * np.sin(self.fire_animation_frame * 0.5))
            glow_color = (255, 100, 0, 100) if self.grid[i, j] == self.FIRE else (255, 50, 50, 120)
            
            # Create glow effect with multiple circles
            for radius_offset in range(3):
                glow_r = glow_radius - radius_offset * 3
                alpha = 50 - radius_offset * 15
                glow_surface = pygame.Surface((glow_r * 2, glow_r * 2))
                glow_surface.set_alpha(alpha)
                glow_surface.fill((255, 80, 0))
                
                glow_rect = glow_surface.get_rect()
                glow_rect.center = (x + self.cell_size//2, y + self.cell_size//2 - height_offset)
                self.screen.blit(glow_surface, glow_rect)
            
            # Draw animated fire particles
            for particle in range(8):
                particle_offset_x = random.randint(-8, 8)
                particle_offset_y = random.randint(-12, -4)
                particle_x = x + self.cell_size//2 + particle_offset_x
                particle_y = y + self.cell_size//2 + particle_offset_y - height_offset
                
                # Animate particles rising
                particle_y -= (self.fire_animation_frame * 2) % 20
                
                # Particle color varies
                particle_colors = [(255, 200, 0), (255, 100, 0), (255, 50, 0), (200, 0, 0)]
                particle_color = random.choice(particle_colors)
                
                # Draw particle
                pygame.draw.circle(self.screen, particle_color, 
                                 (int(particle_x), int(particle_y)), 
                                 random.randint(1, 3))
            
            # Draw intense fire special effects
            if self.grid[i, j] == self.FIRE_INTENSE:
                # Add electric-like effects for intense fires
                center_x = x + self.cell_size//2
                center_y = y + self.cell_size//2 - height_offset
                
                for lightning in range(3):
                    end_x = center_x + random.randint(-15, 15)
                    end_y = center_y + random.randint(-15, 15)
                    pygame.draw.line(self.screen, (255, 255, 100), 
                                   (center_x, center_y), (end_x, end_y), 2)
    
    def _draw_drone(self):
        """Draw the firefighter drone with 3D effects and status indicators."""
        if not self.drone:
            return
        
        drone = self.drone
        row, col = drone.position
        
        # Calculate screen position with elevation
        x = col * self.cell_size + 20
        y = row * self.cell_size + 20
        
        # Get elevation for 3D positioning
        elevation = self._get_elevation(row, col)
        height_offset = int(elevation * self.elevation_scale) + 15  # Drones fly above terrain
        
        screen_x = x + self.cell_size // 2
        screen_y = y + self.cell_size // 2 - height_offset
        
        # Draw flight trail
        if len(drone.trail) > 1:
            trail_points = []
            for i, (trail_row, trail_col) in enumerate(drone.trail):
                trail_x = trail_col * self.cell_size + 20 + self.cell_size // 2
                trail_y = trail_row * self.cell_size + 20 + self.cell_size // 2
                trail_elevation = self._get_elevation(trail_row, trail_col)
                trail_height = int(trail_elevation * self.elevation_scale) + 15
                
                trail_points.append((trail_x, trail_y - trail_height))
            
            # Draw trail with fading effect
            for i in range(1, len(trail_points)):
                alpha = int(255 * (i / len(trail_points)))
                trail_color = (100, 100, 100, alpha)
                pygame.draw.line(self.screen, (150, 150, 150), 
                               trail_points[i-1], trail_points[i], 2)
        
        # Drone color based on current activity
        base_color = (255, 100, 100)  # Red for firefighter
        drone_size = 12
        
        # Modify color based on activity
        if drone.target_fire:
            state_color = (255, 200, 0)  # Yellow when targeting fire
        elif drone.water_level < 5:
            state_color = (200, 200, 200)  # Gray when low on water
        elif drone.energy < 30:
            state_color = (255, 150, 0)  # Orange when low energy
        else:
            state_color = base_color
        
        # Draw drone shadow
        shadow_pos = (screen_x + 3, screen_y + 15)
        pygame.draw.circle(self.screen, (50, 50, 50), shadow_pos, drone_size // 2)
        
        # Draw main drone body (3D effect)
        pygame.draw.circle(self.screen, (100, 100, 100), (screen_x, screen_y + 2), drone_size)
        pygame.draw.circle(self.screen, state_color, (screen_x, screen_y), drone_size)
        pygame.draw.circle(self.screen, (255, 255, 255), (screen_x - 2, screen_y - 2), drone_size // 3)
        
        # Draw rotor blades (animated)
        rotor_angle = (self.step_count * 45) % 360  # Fast rotation
        for angle_offset in [0, 90, 180, 270]:
            angle = math.radians(rotor_angle + angle_offset)
            blade_length = drone_size - 2
            blade_x = screen_x + int(math.cos(angle) * blade_length)
            blade_y = screen_y + int(math.sin(angle) * blade_length)
            pygame.draw.line(self.screen, (200, 200, 200), 
                           (screen_x, screen_y), (blade_x, blade_y), 2)
        
        # Draw status bars
        bar_width = 20
        bar_height = 3
        bar_x = screen_x - bar_width // 2
        bar_y = screen_y + drone_size + 8
        
        # Energy bar
        pygame.draw.rect(self.screen, (100, 100, 100), 
                       (bar_x, bar_y, bar_width, bar_height))
        
        energy_width = int((drone.energy / drone.max_energy) * bar_width)
        energy_color = (0, 255, 0) if drone.energy > 50 else (255, 255, 0) if drone.energy > 20 else (255, 0, 0)
        pygame.draw.rect(self.screen, energy_color, 
                       (bar_x, bar_y, energy_width, bar_height))
        
        # Water level bar
        water_bar_y = bar_y + 6
        pygame.draw.rect(self.screen, (100, 100, 100), 
                       (bar_x, water_bar_y, bar_width, bar_height))
        
        water_width = int((drone.water_level / drone.max_water) * bar_width)
        pygame.draw.rect(self.screen, (0, 100, 255), 
                       (bar_x, water_bar_y, water_width, bar_height))
        
        # Draw detection radius
        detection_radius = drone.detection_radius * self.cell_size
        pygame.draw.circle(self.screen, (100, 150, 255), 
                         (screen_x, screen_y), detection_radius, 1)
        
        # Draw target line if drone has a target
        if drone.target_fire:
            target_x = drone.target_fire[1] * self.cell_size + 20 + self.cell_size // 2
            target_y = drone.target_fire[0] * self.cell_size + 20 + self.cell_size // 2
            target_elevation = self._get_elevation(drone.target_fire[0], drone.target_fire[1])
            target_height = int(target_elevation * self.elevation_scale)
            
            pygame.draw.line(self.screen, (255, 255, 0), 
                           (screen_x, screen_y), 
                           (target_x, target_y - target_height), 2)
        
        # Draw drone ID
        font = pygame.font.Font(None, 16)
        id_text = font.render("F1", True, (255, 255, 255))
        self.screen.blit(id_text, (screen_x - 8, screen_y + drone_size + 5))
    
    def _draw_stats_panel(self):
        """Draw modern-styled statistics panel."""
        stats_x = self.window_size + 30
        panel_width = 300
        
        # Draw semi-transparent background panel
        panel_surface = pygame.Surface((panel_width, self.window_size + 80))
        panel_surface.set_alpha(200)
        panel_surface.fill((20, 25, 35))
        self.screen.blit(panel_surface, (self.window_size + 10, 10))
        
        # Draw panel border
        pygame.draw.rect(self.screen, (100, 150, 200), 
                        (self.window_size + 10, 10, panel_width, self.window_size + 80), 2)
        
        # Title
        title_font = pygame.font.Font(None, 28)
        title_text = title_font.render("🔥 PyroGuard AI", True, (255, 200, 100))
        self.screen.blit(title_text, (stats_x, 25))
        
        subtitle_font = pygame.font.Font(None, 20)
        subtitle_text = subtitle_font.render("Advanced Wildfire Simulation", True, (200, 200, 200))
        self.screen.blit(subtitle_text, (stats_x, 50))
        
        # Calculate terrain statistics
        total_vegetation = (np.sum(self.grid == self.TREE) + 
                          np.sum(self.grid == self.DENSE_FOREST) + 
                          np.sum(self.grid == self.SPARSE_FOREST) +
                          np.sum(self.grid == self.GRASS))
        
        # Main statistics with icons and colors
        stats_font = pygame.font.Font(None, 22)
        y_offset = 85
        
        # Fire statistics (red section)
        fire_stats = [
            ("🔥", f"Active Fires: {np.sum((self.grid == self.FIRE) | (self.grid == self.FIRE_INTENSE))}", (255, 100, 100)),
            ("⚡", f"Intense Fires: {np.sum(self.grid == self.FIRE_INTENSE)}", (255, 50, 50)),
            ("💀", f"Burned Areas: {self.total_trees_burned}", (150, 150, 150)),
        ]
        
        for icon, text, color in fire_stats:
            icon_text = stats_font.render(icon, True, color)
            stat_text = stats_font.render(text, True, color)
            self.screen.blit(icon_text, (stats_x, y_offset))
            self.screen.blit(stat_text, (stats_x + 25, y_offset))
            y_offset += 25
        
        y_offset += 10  # Spacing
        
        # Vegetation statistics (green section)
        veg_stats = [
            ("🌲", f"Total Vegetation: {total_vegetation}", (100, 255, 100)),
            ("🌳", f"Dense Forest: {np.sum(self.grid == self.DENSE_FOREST)}", (0, 150, 0)),
            ("🌿", f"Regular Trees: {np.sum(self.grid == self.TREE)}", (50, 200, 50)),
            ("🍃", f"Sparse Forest: {np.sum(self.grid == self.SPARSE_FOREST)}", (150, 255, 150)),
            ("🌾", f"Grassland: {np.sum(self.grid == self.GRASS)}", (200, 255, 100)),
        ]
        
        for icon, text, color in veg_stats:
            icon_text = stats_font.render(icon, True, color)
            stat_text = stats_font.render(text, True, color)
            self.screen.blit(icon_text, (stats_x, y_offset))
            self.screen.blit(stat_text, (stats_x + 25, y_offset))
            y_offset += 25
        
        y_offset += 10  # Spacing
        
        # Environment statistics (blue section)
        env_stats = [
            ("💧", f"Water Bodies: {np.sum(self.grid == self.WATER)}", (100, 200, 255)),
            ("🗿", f"Rocky Terrain: {np.sum(self.grid == self.ROCK)}", (150, 150, 150)),
            ("⏱️", f"Simulation Step: {self.step_count}", (255, 255, 100)),
        ]
        
        # Add drone statistics if enabled
        if self.enable_drones and self.drone:
            drone_status = self.drone.get_status()
            env_stats.extend([
                ("🚁", f"Firefighter Drone: Active", (255, 100, 100)),
                ("🔥", f"Fires Extinguished: {drone_status['fires_extinguished']}", (255, 200, 0)),
                ("💧", f"Water: {drone_status['water_percentage']:.0f}%", (0, 150, 255)),
                ("⚡", f"Energy: {drone_status['energy_percentage']:.0f}%", (100, 255, 100)),
            ])
        
        for icon, text, color in env_stats:
            icon_text = stats_font.render(icon, True, color)
            stat_text = stats_font.render(text, True, color)
            self.screen.blit(icon_text, (stats_x, y_offset))
            self.screen.blit(stat_text, (stats_x + 25, y_offset))
            y_offset += 25
        
        # Wind indicator with visual compass
        y_offset += 15
        wind_title = stats_font.render("💨 Wind Conditions", True, (255, 255, 150))
        self.screen.blit(wind_title, (stats_x, y_offset))
        y_offset += 30
        
        # Draw wind compass
        compass_center = (stats_x + 60, y_offset + 30)
        compass_radius = 25
        
        # Draw compass circle
        pygame.draw.circle(self.screen, (100, 100, 100), compass_center, compass_radius, 2)
        
        # Draw wind direction arrow
        wind_rad = np.radians(self.wind_direction)
        arrow_end = (
            compass_center[0] + int(compass_radius * 0.8 * np.sin(wind_rad)),
            compass_center[1] - int(compass_radius * 0.8 * np.cos(wind_rad))
        )
        
        # Wind strength affects arrow color
        strength_color_intensity = min(255, int(self.wind_strength * 500))
        arrow_color = (255, 255 - strength_color_intensity, 0)
        
        pygame.draw.line(self.screen, arrow_color, compass_center, arrow_end, 3)
        
        # Draw arrowhead
        arrow_size = 8
        perp_angle1 = wind_rad + np.pi * 0.8
        perp_angle2 = wind_rad - np.pi * 0.8
        
        arrowhead1 = (
            arrow_end[0] + int(arrow_size * np.sin(perp_angle1)),
            arrow_end[1] - int(arrow_size * np.cos(perp_angle1))
        )
        arrowhead2 = (
            arrow_end[0] + int(arrow_size * np.sin(perp_angle2)),
            arrow_end[1] - int(arrow_size * np.cos(perp_angle2))
        )
        
        pygame.draw.polygon(self.screen, arrow_color, [arrow_end, arrowhead1, arrowhead2])
        
        # Wind stats text
        wind_stats = [
            f"Direction: {self.wind_direction:.0f}°",
            f"Strength: {self.wind_strength:.2f}",
        ]
        
        small_font = pygame.font.Font(None, 18)
        for i, stat in enumerate(wind_stats):
            text = small_font.render(stat, True, (200, 200, 200))
            self.screen.blit(text, (stats_x + 130, y_offset + 15 + i * 20))
        
        y_offset += 80
        
        # Enhanced legend with better styling
        legend_title = stats_font.render("🗺️ Terrain Legend", True, (255, 255, 150))
        self.screen.blit(legend_title, (stats_x, y_offset))
        y_offset += 25
        
        legend_items = [
            ("🌲", "Dense Forest", self.COLORS[self.DENSE_FOREST]),
            ("🌳", "Regular Trees", self.COLORS[self.TREE]),
            ("🌿", "Sparse Forest", self.COLORS[self.SPARSE_FOREST]),
            ("🌾", "Grassland", self.COLORS[self.GRASS]),
            ("🔥", "Fire", self.COLORS[self.FIRE]),
            ("⚡", "Intense Fire", self.COLORS[self.FIRE_INTENSE]),
            ("💀", "Burned", self.COLORS[self.BURNED]),
            ("💧", "Water", self.COLORS[self.WATER]),
            ("🗿", "Rock", self.COLORS[self.ROCK])
        ]
        
        legend_font = pygame.font.Font(None, 18)
        for i, (icon, label, color) in enumerate(legend_items):
            y_pos = y_offset + i * 22
            
            # Draw terrain color sample
            color_rect = pygame.Rect(stats_x, y_pos + 2, 16, 16)
            pygame.draw.rect(self.screen, color, color_rect)
            pygame.draw.rect(self.screen, (100, 100, 100), color_rect, 1)
            
            # Draw icon and label
            icon_text = legend_font.render(icon, True, (255, 255, 255))
            label_text = legend_font.render(label, True, (200, 200, 200))
            self.screen.blit(icon_text, (stats_x + 20, y_pos))
            self.screen.blit(label_text, (stats_x + 40, y_pos))
        
        # Controls section
        y_offset += len(legend_items) * 22 + 20
        controls_title = stats_font.render("🎮 Controls", True, (255, 255, 150))
        self.screen.blit(controls_title, (stats_x, y_offset))
        y_offset += 25
        
        control_items = [
            "SPACE - Spawn fires",
            "W - Change wind", 
            "R - Reset forest",
            "Click - Ignite terrain"
        ]
        
        control_font = pygame.font.Font(None, 16)
        for i, control in enumerate(control_items):
            text = control_font.render(control, True, (180, 180, 180))
            self.screen.blit(text, (stats_x, y_offset + i * 18))
    
    def close(self):
        """Clean up pygame resources."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None


def main():
    """Demo the basic wildfire environment."""
    print("🔥 PyroGuard: Basic Wildfire Environment Demo")
    print("=" * 50)
    
    # Create environment optimized for demo
    env = WildfireEnvironment(
        grid_size=18, 
        fire_spread_prob=0.4, 
        initial_tree_density=0.85,
        wind_strength=0.2,
        fire_persistence=4,  # Fires burn for 4 steps
        new_fire_rate=0.12   # More frequent new fires
    )
    
    # Reset and run simulation
    state = env.reset()
    print(f"Initial state: {state['fires_active']} fires, {state['trees_remaining']} trees")
    
    running = True
    step = 0
    
    try:
        while running and not state['is_done'] and step < 500:  # Longer demo
            # Render the environment
            if not env.render():
                break
            
            # Step the simulation
            state = env.step()
            step += 1
            
            # Print periodic updates
            if step % 10 == 0:
                print(f"Step {step}: {state['fires_active']} fires, "
                      f"{state['trees_remaining']} trees remaining, "
                      f"{state['trees_burned']} burned")
            
            # Check for termination
            if state['fires_active'] == 0:
                print("🎉 All fires extinguished!")
                break
            elif state['trees_remaining'] == 0:
                print("💀 All trees burned!")
                break
        
        # Final statistics
        print("\n" + "=" * 50)
        print("SIMULATION COMPLETE")
        print(f"Total steps: {state['step']}")
        print(f"Total trees burned: {state['total_trees_burned']}")
        print(f"Total fires started: {state['total_fires_started']}")
        print(f"Final trees remaining: {state['trees_remaining']}")
        
        # Keep window open for a bit
        import time
        time.sleep(3)
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        env.close()


if __name__ == "__main__":
    main()
