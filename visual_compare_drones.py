#!/usr/bin/env python3
"""
Visual Comparison: RL Drone vs Random Drone
Shows both drones working on identical fire scenarios with side-by-side visualization.
"""

import numpy as np
import random
import pygame
import time
from stable_baselines3 import PPO
from wildfire_env import WildfireEnvironment
from simple_drone import FirefighterDrone


class VisualComparison:
    """Visual comparison of RL drone vs Random drone."""
    
    def __init__(self, grid_size=25):
        self.grid_size = grid_size
        self.cell_size = 25
        self.window_width = grid_size * self.cell_size * 2 + 100  # Side by side
        self.window_height = grid_size * self.cell_size + 200
        self.screen = None
        self.clock = None
        
        # Colors
        self.COLORS = {
            0: (139, 69, 19),      # Empty - Brown
            1: (34, 139, 34),      # Tree - Green
            2: (255, 69, 0),       # Fire - Red
            3: (64, 64, 64),       # Burned - Gray
            4: (0, 191, 255),      # Water - Blue
            5: (255, 0, 0),        # Fire Intense - Bright Red
            6: (0, 100, 0),        # Dense Forest - Dark Green
            7: (144, 238, 144),    # Sparse Forest - Light Green
            8: (105, 105, 105),    # Rock - Gray
            9: (154, 205, 50)      # Grass - Yellow Green
        }
        
        # Initialize environments
        self.env_rl = None
        self.env_random = None
        self.rl_model = None
        self.fire_locations = []
        
        # Results tracking
        self.results = {
            'rl': {'fires_extinguished': 0, 'steps': 0, 'energy_used': 0, 'trees_burned': 0},
            'random': {'fires_extinguished': 0, 'steps': 0, 'energy_used': 0, 'trees_burned': 0}
        }
    
    def initialize(self):
        """Initialize pygame and environments."""
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("🔥 PyroGuard AI - RL vs Random Drone Comparison")
        self.clock = pygame.time.Clock()
        
        # Load RL model - using the BEST model (not final)
        try:
            self.rl_model = PPO.load("models/best_model/best_model.zip")
            print("✅ Best RL model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading RL model: {e}")
            return False
        
        # Generate fire scenario
        self.fire_locations = self._generate_fire_scenario()
        
        # Initialize environments
        self.env_rl = WildfireEnvironment(
            grid_size=self.grid_size,
            fire_spread_prob=0.1,
            wind_strength=0.05,
            fire_persistence=8,
            new_fire_rate=0.0,  # No random fires
            enable_drones=True
        )
        
        self.env_random = WildfireEnvironment(
            grid_size=self.grid_size,
            fire_spread_prob=0.1,
            wind_strength=0.05,
            fire_persistence=8,
            new_fire_rate=0.0,  # No random fires
            enable_drones=True
        )
        
        # CRITICAL: Ensure both drones start at the EXACT same position
        # Set both drones to start at base (0, 0)
        if self.env_rl.drone:
            self.env_rl.drone.position = [0, 0]
        if self.env_random.drone:
            self.env_random.drone.position = [0, 0]
        
        # VERIFY: Both drones start at identical positions
        print(f"🔍 RL Drone starting position: {self.env_rl.drone.position if self.env_rl.drone else 'None'}")
        print(f"🔍 Random Drone starting position: {self.env_random.drone.position if self.env_random.drone else 'None'}")
        
        # CRITICAL: Force synchronization of starting positions
        if self.env_rl.drone and self.env_random.drone:
            self.env_random.drone.position = self.env_rl.drone.position.copy()
            print(f"✅ Synchronized both drones to: {self.env_rl.drone.position}")
        
        # Set identical fire scenarios
        self._set_identical_fires()
        
        return True
    
    def _generate_fire_scenario(self):
        """Generate exactly 2 fires that are not adjacent to each other."""
        fire_locations = []
        attempts = 0
        
        # Generate exactly 2 fires
        while len(fire_locations) < 2 and attempts < 100:
            # Spread fires more widely across the grid
            row = random.randint(1, self.grid_size - 2)
            col = random.randint(1, self.grid_size - 2)
            
            # Ensure minimum distance between fires (at least 3 cells apart)
            too_close = False
            for existing_fire in fire_locations:
                distance = ((row - existing_fire[0])**2 + (col - existing_fire[1])**2)**0.5
                if distance < 3:
                    too_close = True
                    break
            
            if not too_close and (row, col) not in fire_locations:
                fire_locations.append((row, col))
            attempts += 1
        
        print(f"🔥 Generated exactly {len(fire_locations)} fires (not adjacent): {fire_locations}")
        return fire_locations
    
    def _set_identical_fires(self):
        """Set identical fires in both environments."""
        # Reset both environments
        self.env_rl.reset()
        self.env_random.reset()
        
        # CRITICAL: Make both environments completely identical
        # Copy the entire grid state from one to the other
        self.env_random.grid = self.env_rl.grid.copy()
        self.env_random.fire_age = self.env_rl.fire_age.copy()
        
        # Set single-tree fires at identical locations
        for fire_pos in self.fire_locations:
            if (0 <= fire_pos[0] < self.grid_size and 
                0 <= fire_pos[1] < self.grid_size):
                # Ensure the cell is a tree before setting it on fire
                if self.env_rl.grid[fire_pos[0], fire_pos[1]] in [self.env_rl.TREE, self.env_rl.DENSE_FOREST, self.env_rl.SPARSE_FOREST]:
                    # Set fire in both environments
                    self.env_rl.grid[fire_pos[0], fire_pos[1]] = self.env_rl.FIRE
                    self.env_rl.fire_age[fire_pos[0], fire_pos[1]] = 0
                    
                    self.env_random.grid[fire_pos[0], fire_pos[1]] = self.env_random.FIRE
                    self.env_random.fire_age[fire_pos[0], fire_pos[1]] = 0
                    print(f"🔥 Set fire at {fire_pos}")
                else:
                    print(f"⚠️  Position {fire_pos} is not a tree, skipping fire placement")
    
    def run_comparison(self, max_steps=300):
        """Run the visual comparison."""
        if not self.initialize():
            return
        
        print("🚁 Starting Visual Comparison...")
        print(f"Fire locations: {self.fire_locations}")
        print("Press SPACE to step, ESC to exit")
        
        step = 0
        running = True
        
        while running and step < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # Manual step
                        self._step_both_drones()
                        step += 1
                    elif event.key == pygame.K_a:
                        # Auto mode
                        self._auto_run()
                        step += 50
            
            # Auto step every few frames
            if step % 3 == 0:
                self._step_both_drones()
                step += 1
            
            # Render
            self._render()
            self.clock.tick(10)  # 10 FPS
            
            # Check if both are done
            rl_done = self._is_done(self.env_rl)
            random_done = self._is_done(self.env_random)
            
            if rl_done and random_done:
                print("🎉 Both drones completed their missions!")
                break
        
        # Show final results
        self._show_final_results()
        
        # Keep window open
        print("Press any key to exit...")
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    waiting = False
            self._render()
            self.clock.tick(10)
        
        pygame.quit()
    
    def _step_both_drones(self):
        """Step both drone environments."""
        # CRITICAL: Ensure both drones start each step at the SAME position
        if self.env_rl.drone and self.env_random.drone:
            # Force both drones to start at the same position each step
            self.env_random.drone.position = self.env_rl.drone.position.copy()
        
        # REMOVED: Other drone state synchronization - this was making the comparison unfair!
        # Both drones should operate independently with their own decisions and consequences
        
        # Use proper environment step functions that handle fire suppression correctly
        # RL drone environment
        if self.env_rl.drone:
            # Get RL observation and action
            obs = self._get_rl_observation()
            if obs is not None and self.rl_model:
                action, _ = self.rl_model.predict(obs, deterministic=True)
                # DEBUG: Print action to verify RL model is being used
                if self.results['rl']['steps'] < 5:  # Only print first few steps
                    print(f"RL Action {self.results['rl']['steps']}: {action}")
                # Apply RL action to drone
                self._apply_rl_action_to_drone(action)
        
        # Random drone environment  
        if self.env_random.drone:
            # Make random decision
            self._make_random_drone_decisions()
            # Apply random action
            action = random.randint(0, 5)
            self._apply_random_action_to_drone(action)
        
        # Step both environments (this handles fire suppression properly)
        self.env_rl.step()
        self.env_random.step()
        
        # REMOVED: Grid synchronization - this was making fire spread identical!
        # Each environment should have independent fire spread based on drone actions
        
        # Update results
        self._update_results()
    
    def _get_rl_observation(self):
        """Get observation for RL drone."""
        if not self.env_rl.drone:
            return None
        
        drone = self.env_rl.drone
        row, col = drone.position
        
        # Extract 5x5 grid around drone
        grid_obs = np.zeros((5, 5), dtype=np.float32)
        for i in range(5):
            for j in range(5):
                obs_row = row - 2 + i
                obs_col = col - 2 + j
                
                if 0 <= obs_row < self.grid_size and 0 <= obs_col < self.grid_size:
                    grid_obs[i, j] = float(self.env_rl.grid[obs_row, obs_col])
                else:
                    grid_obs[i, j] = -1.0
        
        # Flatten grid
        grid_flat = grid_obs.flatten()
        
        # Drone status features
        status_features = np.array([
            drone.water_level / drone.max_water,
            drone.energy / drone.max_energy,
            min(drone.fires_extinguished / 10.0, 1.0),
            min(len(drone.known_fires) / 5.0, 1.0),
            1.0 if drone.refueling_time > 0 else 0.0,
            self._get_distance_to_nearest_fire(drone) / 10.0
        ], dtype=np.float32)
        
        return np.concatenate([grid_flat, status_features])
    
    def _apply_rl_action_to_drone(self, action):
        """Apply RL action to drone with full fire suppression capabilities."""
        if not self.env_rl.drone:
            return
        
        drone = self.env_rl.drone
        
        # First, let the drone scan for fires and update its state
        detected_fires = drone._scan_for_fires(self.env_rl.grid)
        drone._update_known_fires(detected_fires, self.env_rl.grid)
        
        # Apply the RL action
        if action == 0:  # Move North
            new_pos = (drone.position[0] - 1, drone.position[1])
        elif action == 1:  # Move South
            new_pos = (drone.position[0] + 1, drone.position[1])
        elif action == 2:  # Move East
            new_pos = (drone.position[0], drone.position[1] + 1)
        elif action == 3:  # Move West
            new_pos = (drone.position[0], drone.position[1] - 1)
        elif action == 4:  # Scan (stay in place)
            new_pos = drone.position
        elif action == 5:  # Suppress fire
            new_pos = drone.position
            # Try to suppress nearby fires
            suppression_result = drone._try_suppress_fire(self.env_rl.grid)
            if suppression_result['action'] == 'fire_suppressed':
                return suppression_result
        else:
            new_pos = drone.position
        
        # Apply movement if valid
        if (0 <= new_pos[0] < self.grid_size and 
            0 <= new_pos[1] < self.grid_size):
            drone.position = list(new_pos)
        
        # Update drone's internal state
        drone.steps_taken += 1
        drone.energy = max(0, drone.energy - 1)
        
        # Update trail
        drone.trail.append(tuple(drone.position))
        if len(drone.trail) > drone.max_trail_length:
            drone.trail.pop(0)
        
        return {'action': 'move', 'position': drone.position}
    
    def _apply_rl_action(self, action):
        """Apply RL action to drone."""
        if not self.env_rl.drone:
            return
        
        drone = self.env_rl.drone
        
        if action == 0:  # Move North
            new_pos = (drone.position[0] - 1, drone.position[1])
        elif action == 1:  # Move South
            new_pos = (drone.position[0] + 1, drone.position[1])
        elif action == 2:  # Move East
            new_pos = (drone.position[0], drone.position[1] + 1)
        elif action == 3:  # Move West
            new_pos = (drone.position[0], drone.position[1] - 1)
        else:
            new_pos = drone.position
        
        # Apply movement
        if (0 <= new_pos[0] < self.grid_size and 
            0 <= new_pos[1] < self.grid_size):
            drone.position = list(new_pos)
    
    def _make_random_drone_decisions(self):
        """Make the random drone make random decisions instead of intelligent ones."""
        if not self.env_random.drone:
            return
        
        drone = self.env_random.drone
        
        # Randomly choose target fire from known fires
        if drone.known_fires and random.random() < 0.3:  # 30% chance to change target
            drone.target_fire = random.choice(drone.known_fires)
        
        # Randomly choose patrol target
        if not drone.target_fire and random.random() < 0.2:  # 20% chance to set new patrol
            drone.patrol_target = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1)
            )
    
    def _apply_random_action_to_drone(self, action):
        """Apply random action to drone with proper fire suppression."""
        if not self.env_random.drone:
            return
        
        drone = self.env_random.drone
        
        if action == 0:  # Move North
            new_pos = (drone.position[0] - 1, drone.position[1])
        elif action == 1:  # Move South
            new_pos = (drone.position[0] + 1, drone.position[1])
        elif action == 2:  # Move East
            new_pos = (drone.position[0], drone.position[1] + 1)
        elif action == 3:  # Move West
            new_pos = (drone.position[0], drone.position[1] - 1)
        elif action == 4:  # Scan (stay in place)
            new_pos = drone.position
        elif action == 5:  # Suppress fire
            new_pos = drone.position
            # Try to suppress nearby fires
            suppression_result = drone._try_suppress_fire(self.env_random.grid)
            if suppression_result['action'] == 'fire_suppressed':
                return suppression_result
        else:
            new_pos = drone.position
        
        # Apply movement if valid
        if (0 <= new_pos[0] < self.grid_size and 
            0 <= new_pos[1] < self.grid_size):
            drone.position = list(new_pos)
        
        # Update drone's internal state
        drone.steps_taken += 1
        drone.energy = max(0, drone.energy - 1)
        
        # Update trail
        drone.trail.append(tuple(drone.position))
        if len(drone.trail) > drone.max_trail_length:
            drone.trail.pop(0)
        
        return {'action': 'move', 'position': drone.position}
    
    def _apply_random_action(self, action):
        """Apply random action to drone."""
        if not self.env_random.drone:
            return
        
        drone = self.env_random.drone
        
        if action == 0:  # Move North
            new_pos = (drone.position[0] - 1, drone.position[1])
        elif action == 1:  # Move South
            new_pos = (drone.position[0] + 1, drone.position[1])
        elif action == 2:  # Move East
            new_pos = (drone.position[0], drone.position[1] + 1)
        elif action == 3:  # Move West
            new_pos = (drone.position[0], drone.position[1] - 1)
        else:
            new_pos = drone.position
        
        # Apply movement
        if (0 <= new_pos[0] < self.grid_size and 
            0 <= new_pos[1] < self.grid_size):
            drone.position = list(new_pos)
    
    def _get_distance_to_nearest_fire(self, drone):
        """Get distance to nearest fire."""
        if not drone.known_fires:
            return 10.0
        
        min_distance = float('inf')
        for fire_pos in drone.known_fires:
            distance = np.sqrt((drone.position[0] - fire_pos[0])**2 + 
                            (drone.position[1] - fire_pos[1])**2)
            min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 10.0
    
    def _is_done(self, env):
        """Check if environment is done."""
        current_fires = np.sum((env.grid == env.FIRE) | (env.grid == env.FIRE_INTENSE))
        return current_fires == 0
    
    def _update_results(self):
        """Update results tracking."""
        # RL results
        if self.env_rl.drone:
            self.results['rl']['fires_extinguished'] = self.env_rl.drone.fires_extinguished
            self.results['rl']['steps'] += 1
            self.results['rl']['energy_used'] = 100 - self.env_rl.drone.energy
        
        # Random results
        if self.env_random.drone:
            self.results['random']['fires_extinguished'] = self.env_random.drone.fires_extinguished
            self.results['random']['steps'] += 1
            self.results['random']['energy_used'] = 100 - self.env_random.drone.energy
        
        # Count burned trees
        self.results['rl']['trees_burned'] = np.sum(self.env_rl.grid == self.env_rl.BURNED)
        self.results['random']['trees_burned'] = np.sum(self.env_random.grid == self.env_random.BURNED)
    
    def _render(self):
        """Render both environments side by side."""
        self.screen.fill((50, 50, 50))  # Dark background
        
        # Render RL environment (left side)
        self._render_environment(self.env_rl, 0, "RL Drone")
        
        # Render Random environment (right side)
        self._render_environment(self.env_random, self.grid_size * self.cell_size + 50, "Random Drone")
        
        # Render results panel
        self._render_results_panel()
        
        pygame.display.flip()
    
    def _render_environment(self, env, x_offset, title):
        """Render a single environment."""
        # Draw title
        font = pygame.font.Font(None, 24)
        title_surface = font.render(title, True, (255, 255, 255))
        self.screen.blit(title_surface, (x_offset, 10))
        
        # Draw grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = x_offset + j * self.cell_size
                y = 40 + i * self.cell_size
                
                cell_type = env.grid[i, j]
                color = self.COLORS.get(cell_type, (100, 100, 100))
                
                pygame.draw.rect(self.screen, color, (x, y, self.cell_size-1, self.cell_size-1))
        
        # Draw drone
        if env.drone:
            drone_x = x_offset + env.drone.position[1] * self.cell_size + self.cell_size // 2
            drone_y = 40 + env.drone.position[0] * self.cell_size + self.cell_size // 2
            
            # Drone color based on type
            drone_color = (255, 100, 100) if "RL" in title else (100, 255, 100)
            pygame.draw.circle(self.screen, drone_color, (drone_x, drone_y), 8)
            
            # Draw trail
            if hasattr(env.drone, 'trail') and len(env.drone.trail) > 1:
                for i, (trail_row, trail_col) in enumerate(env.drone.trail):
                    trail_x = x_offset + trail_col * self.cell_size + self.cell_size // 2
                    trail_y = 40 + trail_row * self.cell_size + self.cell_size // 2
                    alpha = int(255 * (i / len(env.drone.trail)))
                    trail_color = (*drone_color[:3], alpha)
                    pygame.draw.circle(self.screen, drone_color, (trail_x, trail_y), 3)
    
    def _render_results_panel(self):
        """Render results panel at bottom."""
        panel_y = self.grid_size * self.cell_size + 50
        panel_height = 150
        
        # Draw panel background
        pygame.draw.rect(self.screen, (30, 30, 30), (0, panel_y, self.window_width, panel_height))
        
        # Draw results
        font = pygame.font.Font(None, 20)
        y_offset = panel_y + 10
        
        # RL Results
        rl_text = [
            f"RL Drone:",
            f"  Fires Extinguished: {self.results['rl']['fires_extinguished']}",
            f"  Steps Taken: {self.results['rl']['steps']}",
            f"  Energy Used: {self.results['rl']['energy_used']}",
            f"  Trees Burned: {self.results['rl']['trees_burned']}"
        ]
        
        for i, text in enumerate(rl_text):
            color = (255, 100, 100) if i == 0 else (255, 255, 255)
            text_surface = font.render(text, True, color)
            self.screen.blit(text_surface, (10, y_offset + i * 20))
        
        # Random Results
        random_text = [
            f"Random Drone:",
            f"  Fires Extinguished: {self.results['random']['fires_extinguished']}",
            f"  Steps Taken: {self.results['random']['steps']}",
            f"  Energy Used: {self.results['random']['energy_used']}",
            f"  Trees Burned: {self.results['random']['trees_burned']}"
        ]
        
        for i, text in enumerate(random_text):
            color = (100, 255, 100) if i == 0 else (255, 255, 255)
            text_surface = font.render(text, True, color)
            self.screen.blit(text_surface, (self.window_width // 2 + 10, y_offset + i * 20))
        
        # Controls
        controls_text = [
            "Controls:",
            "SPACE - Step manually",
            "A - Auto run",
            "ESC - Exit"
        ]
        
        for i, text in enumerate(controls_text):
            text_surface = font.render(text, True, (200, 200, 200))
            self.screen.blit(text_surface, (10, y_offset + 120 + i * 15))
    
    def _show_final_results(self):
        """Show final comparison results."""
        print("\n" + "="*60)
        print("🏆 FINAL COMPARISON RESULTS")
        print("="*60)
        
        rl = self.results['rl']
        random = self.results['random']
        
        print(f"{'Metric':<25} {'RL Drone':<15} {'Random Drone':<15} {'Improvement':<15}")
        print("-" * 70)
        
        # Fires extinguished
        fires_imp = ((rl['fires_extinguished'] - random['fires_extinguished']) / 
                    max(random['fires_extinguished'], 1) * 100)
        print(f"{'Fires Extinguished':<25} {rl['fires_extinguished']:<15} {random['fires_extinguished']:<15} {fires_imp:<15.1f}%")
        
        # Steps taken
        steps_imp = ((random['steps'] - rl['steps']) / max(rl['steps'], 1) * 100)
        print(f"{'Steps Taken':<25} {rl['steps']:<15} {random['steps']:<15} {steps_imp:<15.1f}%")
        
        # Energy efficiency
        rl_efficiency = rl['fires_extinguished'] / max(rl['energy_used'], 1)
        random_efficiency = random['fires_extinguished'] / max(random['energy_used'], 1)
        eff_imp = ((rl_efficiency - random_efficiency) / max(random_efficiency, 0.1) * 100)
        print(f"{'Energy Efficiency':<25} {rl_efficiency:<15.2f} {random_efficiency:<15.2f} {eff_imp:<15.1f}%")
        
        # Trees burned
        trees_imp = ((random['trees_burned'] - rl['trees_burned']) / max(rl['trees_burned'], 1) * 100)
        print(f"{'Trees Burned':<25} {rl['trees_burned']:<15} {random['trees_burned']:<15} {trees_imp:<15.1f}%")
        
        print("\n🎯 Overall Assessment:")
        if fires_imp > 20 and steps_imp > 10:
            print("✅ EXCELLENT: RL drone significantly outperforms random")
        elif fires_imp > 10:
            print("✅ GOOD: RL drone shows clear improvement")
        elif fires_imp > 0:
            print("⚠️  WEAK: RL drone shows minimal improvement")
        else:
            print("❌ POOR: RL drone performs worse than random")
    
    def _auto_run(self):
        """Auto run for faster comparison."""
        for _ in range(50):
            self._step_both_drones()
            time.sleep(0.01)  # Small delay for visualization


def main():
    """Run the visual comparison."""
    print("🔥 PyroGuard AI - Visual Drone Comparison")
    print("=" * 50)
    
    comparison = VisualComparison(grid_size=25)
    comparison.run_comparison(max_steps=300)


if __name__ == "__main__":
    main()
