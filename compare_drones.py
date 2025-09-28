#!/usr/bin/env python3
"""
Fair Comparison: RL Drone vs Random Drone
Both drones face identical fire scenarios for fair evaluation.
"""

import numpy as np
import random
from stable_baselines3 import PPO
from rl_wildfire_env import RLWildfireEnv
from wildfire_env import WildfireEnvironment
from simple_drone import FirefighterDrone
import time


class ControlledFireEnvironment:
    """Environment that spawns fires in predetermined locations."""
    
    def __init__(self, grid_size=30, fire_locations=None):
        self.grid_size = grid_size
        self.fire_locations = fire_locations or []
        self.base_env = WildfireEnvironment(
            grid_size=grid_size,
            fire_spread_prob=0.1,
            initial_tree_density=0.75,
            wind_strength=0.05,
            fire_persistence=8,
            new_fire_rate=0.0,  # No random fires
            enable_drones=False  # We'll add drone manually
        )
        self.drone = None
        self.step_count = 0
        self.max_steps = 200
        
    def reset_with_fires(self, fire_locations):
        """Reset environment with specific fire locations."""
        self.fire_locations = fire_locations
        state = self.base_env.reset()
        
        # Spawn fires at predetermined locations
        for fire_pos in fire_locations:
            if (0 <= fire_pos[0] < self.grid_size and 
                0 <= fire_pos[1] < self.grid_size):
                self.base_env.grid[fire_pos[0], fire_pos[1]] = self.base_env.FIRE
                self.base_env.fire_age[fire_pos[0], fire_pos[1]] = 0
        
        # Initialize drone
        self.drone = FirefighterDrone((1, 1), self.grid_size)
        self.step_count = 0
        
        return self.get_state()
    
    def step(self):
        """Step the environment."""
        self.step_count += 1
        
        # Update drone
        drone_action = self.drone.update(self.base_env.grid)
        
        # Process drone action
        if drone_action.get('action') == 'fire_suppressed':
            pos = drone_action.get('position')
            if pos and self._is_valid_position(pos):
                if self.base_env.grid[pos[0], pos[1]] in [self.base_env.FIRE, self.base_env.FIRE_INTENSE]:
                    self.base_env.grid[pos[0], pos[1]] = self.base_env.BURNED
                    self.base_env.fire_age[pos[0], pos[1]] = 0
        
        # Step base environment (fire spread)
        state = self.base_env.step()
        
        # Check termination
        current_fires = np.sum((self.base_env.grid == self.base_env.FIRE) | 
                             (self.base_env.grid == self.base_env.FIRE_INTENSE))
        terminated = current_fires == 0
        truncated = self.step_count >= self.max_steps
        
        return {
            'fires_active': current_fires,
            'fires_extinguished': self.drone.fires_extinguished,
            'drone_energy': self.drone.energy,
            'drone_water': self.drone.water_level,
            'step': self.step_count,
            'terminated': terminated,
            'truncated': truncated
        }
    
    def get_state(self):
        """Get current state."""
        current_fires = np.sum((self.base_env.grid == self.base_env.FIRE) | 
                             (self.base_env.grid == self.base_env.FIRE_INTENSE))
        return {
            'fires_active': current_fires,
            'fires_extinguished': self.drone.fires_extinguished if self.drone else 0,
            'drone_energy': self.drone.energy if self.drone else 100,
            'drone_water': self.drone.water_level if self.drone else 20,
            'step': self.step_count
        }
    
    def _is_valid_position(self, pos):
        """Check if position is valid."""
        row, col = pos
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size


class RandomDrone:
    """Random drone that takes random actions."""
    
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.position = [1, 1]  # Start at base
        self.energy = 100
        self.water_level = 20
        self.fires_extinguished = 0
        self.steps_taken = 0
        
    def update(self, environment_grid):
        """Take random action."""
        self.steps_taken += 1
        self.energy = max(0, self.energy - 1)
        
        # Random action: 0-3 for movement, 4 for scan, 5 for suppress
        action = random.randint(0, 5)
        
        if action == 0:  # Move North
            new_pos = (self.position[0] - 1, self.position[1])
        elif action == 1:  # Move South
            new_pos = (self.position[0] + 1, self.position[1])
        elif action == 2:  # Move East
            new_pos = (self.position[0], self.position[1] + 1)
        elif action == 3:  # Move West
            new_pos = (self.position[0], self.position[1] - 1)
        else:
            new_pos = self.position  # Stay in place for scan/suppress
        
        # Check if move is valid
        if (0 <= new_pos[0] < self.grid_size and 
            0 <= new_pos[1] < self.grid_size):
            self.position = list(new_pos)
        
        # Try to suppress fires if action is 5
        if action == 5 and self.water_level > 0:
            row, col = self.position
            # Check adjacent cells for fires
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    target_row, target_col = row + dr, col + dc
                    if (0 <= target_row < self.grid_size and 
                        0 <= target_col < self.grid_size):
                        if environment_grid[target_row, target_col] in [2, 5]:  # FIRE or FIRE_INTENSE
                            self.water_level = max(0, self.water_level - 2)
                            self.fires_extinguished += 1
                            return {
                                'action': 'fire_suppressed',
                                'position': (target_row, target_col),
                                'water_remaining': self.water_level,
                                'fires_extinguished': self.fires_extinguished
                            }
        
        # Return to base if low energy or water
        if self.energy < 20 or self.water_level == 0:
            if self.position == [0, 0]:
                self.energy = 100
                self.water_level = 20
                return {'action': 'refueled'}
            else:
                # Move towards base
                if self.position[0] > 0:
                    self.position[0] -= 1
                elif self.position[1] > 0:
                    self.position[1] -= 1
        
        return {'action': 'move', 'position': self.position}


def generate_fire_scenarios(num_scenarios=10, grid_size=30, num_fires=8):
    """Generate random fire scenarios for testing."""
    scenarios = []
    
    for _ in range(num_scenarios):
        # Generate random fire locations
        fire_locations = []
        attempts = 0
        while len(fire_locations) < num_fires and attempts < 100:
            row = random.randint(2, grid_size - 3)  # Avoid edges
            col = random.randint(2, grid_size - 3)
            if (row, col) not in fire_locations:
                fire_locations.append((row, col))
            attempts += 1
        
        scenarios.append(fire_locations)
    
    return scenarios


def run_drone_simulation(env, drone_type, scenario_id, max_steps=200):
    """Run simulation for a specific drone type."""
    results = {
        'scenario_id': scenario_id,
        'drone_type': drone_type,
        'total_reward': 0,
        'fires_extinguished': 0,
        'steps_taken': 0,
        'energy_used': 0,
        'success': False,
        'final_fires': 0
    }
    
    initial_energy = 100
    step = 0
    
    while step < max_steps:
        if drone_type == 'RL':
            # RL drone uses trained model
            obs = env._get_observation() if hasattr(env, '_get_observation') else None
            if obs is not None:
                action, _ = rl_model.predict(obs, deterministic=True)
                # Convert RL action to drone action
                if action == 0:  # North
                    new_pos = (env.drone.position[0] - 1, env.drone.position[1])
                elif action == 1:  # South
                    new_pos = (env.drone.position[0] + 1, env.drone.position[1])
                elif action == 2:  # East
                    new_pos = (env.drone.position[0], env.drone.position[1] + 1)
                elif action == 3:  # West
                    new_pos = (env.drone.position[0], env.drone.position[1] - 1)
                else:
                    new_pos = env.drone.position
                
                # Apply movement
                if (0 <= new_pos[0] < env.grid_size and 
                    0 <= new_pos[1] < env.grid_size):
                    env.drone.position = list(new_pos)
        
        # Step environment
        state = env.step()
        step += 1
        
        # Update results
        results['fires_extinguished'] = state['fires_extinguished']
        results['steps_taken'] = step
        results['energy_used'] = initial_energy - state['drone_energy']
        results['final_fires'] = state['fires_active']
        
        # Check termination
        if state['terminated']:
            results['success'] = True
            break
        elif state['truncated']:
            break
    
    # Calculate reward (simple reward function)
    results['total_reward'] = results['fires_extinguished'] * 10 - results['steps_taken'] * 0.1
    
    return results


def compare_drones(num_scenarios=10, grid_size=30, num_fires=8):
    """Compare RL drone vs Random drone on identical scenarios."""
    print("🔥 PyroGuard AI - Fair Drone Comparison")
    print("=" * 50)
    print(f"Testing {num_scenarios} scenarios with {num_fires} fires each")
    print("Both drones face identical fire scenarios")
    print()
    
    # Load RL model
    global rl_model
    try:
        rl_model = PPO.load("models/final_model_ppo.zip")
        print("✅ RL model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading RL model: {e}")
        return
    
    # Generate fire scenarios
    scenarios = generate_fire_scenarios(num_scenarios, grid_size, num_fires)
    
    # Results storage
    rl_results = []
    random_results = []
    
    print("🚁 Running comparisons...")
    print()
    
    for i, fire_locations in enumerate(scenarios):
        print(f"Scenario {i+1}/{num_scenarios}: {len(fire_locations)} fires")
        
        # Test RL Drone
        env_rl = ControlledFireEnvironment(grid_size)
        env_rl.reset_with_fires(fire_locations)
        
        # Replace drone with RL-compatible one
        from rl_wildfire_env import RLWildfireEnv
        rl_env = RLWildfireEnv(grid_size, max_steps=200)
        rl_env.base_env.grid = env_rl.base_env.grid.copy()
        rl_env.base_env.fire_age = env_rl.base_env.fire_age.copy()
        
        # Run RL drone
        rl_result = run_rl_simulation(rl_env, i)
        rl_results.append(rl_result)
        
        # Test Random Drone
        env_random = ControlledFireEnvironment(grid_size)
        env_random.reset_with_fires(fire_locations)
        env_random.drone = RandomDrone(grid_size)
        
        random_result = run_random_simulation(env_random, i)
        random_results.append(random_result)
        
        print(f"  RL: {rl_result['fires_extinguished']} fires, {rl_result['steps_taken']} steps, "
              f"Reward: {rl_result['total_reward']:.1f}")
        print(f"  Random: {random_result['fires_extinguished']} fires, {random_result['steps_taken']} steps, "
              f"Reward: {random_result['total_reward']:.1f}")
        print()
    
    # Calculate statistics
    rl_avg_reward = np.mean([r['total_reward'] for r in rl_results])
    random_avg_reward = np.mean([r['total_reward'] for r in random_results])
    
    rl_avg_fires = np.mean([r['fires_extinguished'] for r in rl_results])
    random_avg_fires = np.mean([r['fires_extinguished'] for r in random_results])
    
    rl_avg_steps = np.mean([r['steps_taken'] for r in rl_results])
    random_avg_steps = np.mean([r['steps_taken'] for r in random_results])
    
    rl_success_rate = np.mean([r['success'] for r in rl_results])
    random_success_rate = np.mean([r['success'] for r in random_results])
    
    # Print results
    print("📊 COMPARISON RESULTS:")
    print("=" * 50)
    print(f"{'Metric':<20} {'RL Drone':<15} {'Random Drone':<15} {'Improvement':<15}")
    print("-" * 65)
    print(f"{'Avg Reward':<20} {rl_avg_reward:<15.1f} {random_avg_reward:<15.1f} {((rl_avg_reward - random_avg_reward) / random_avg_reward * 100):<15.1f}%")
    print(f"{'Avg Fires Extinguished':<20} {rl_avg_fires:<15.1f} {random_avg_fires:<15.1f} {((rl_avg_fires - random_avg_fires) / random_avg_fires * 100):<15.1f}%")
    print(f"{'Avg Steps':<20} {rl_avg_steps:<15.1f} {random_avg_steps:<15.1f} {((rl_avg_steps - random_avg_steps) / random_avg_steps * 100):<15.1f}%")
    print(f"{'Success Rate':<20} {rl_success_rate:<15.1%} {random_success_rate:<15.1%} {((rl_success_rate - random_success_rate) / random_success_rate * 100):<15.1f}%")
    
    improvement = ((rl_avg_reward - random_avg_reward) / random_avg_reward * 100)
    print()
    print(f"🎯 Overall Performance Improvement: {improvement:.1f}%")
    
    if improvement > 20:
        print("✅ EXCELLENT: RL drone significantly outperforms random")
    elif improvement > 10:
        print("✅ GOOD: RL drone shows clear improvement")
    elif improvement > 0:
        print("⚠️  WEAK: RL drone shows minimal improvement")
    else:
        print("❌ POOR: RL drone performs worse than random")


def run_rl_simulation(env, scenario_id):
    """Run RL drone simulation."""
    obs, _ = env.reset()
    total_reward = 0
    fires_extinguished = 0
    step = 0
    
    while step < 200:
        action, _ = rl_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        fires_extinguished = info['fires_extinguished']
        step += 1
        
        if terminated or truncated:
            break
    
    return {
        'scenario_id': scenario_id,
        'total_reward': total_reward,
        'fires_extinguished': fires_extinguished,
        'steps_taken': step,
        'success': terminated
    }


def run_random_simulation(env, scenario_id):
    """Run random drone simulation."""
    total_reward = 0
    fires_extinguished = 0
    step = 0
    
    while step < 200:
        # Random action
        action = random.randint(0, 5)
        
        # Apply action to drone
        if action == 0:  # North
            new_pos = (env.drone.position[0] - 1, env.drone.position[1])
        elif action == 1:  # South
            new_pos = (env.drone.position[0] + 1, env.drone.position[1])
        elif action == 2:  # East
            new_pos = (env.drone.position[0], env.drone.position[1] + 1)
        elif action == 3:  # West
            new_pos = (env.drone.position[0], env.drone.position[1] - 1)
        else:
            new_pos = env.drone.position
        
        # Apply movement
        if (0 <= new_pos[0] < env.grid_size and 
            0 <= new_pos[1] < env.grid_size):
            env.drone.position = list(new_pos)
        
        # Step environment
        state = env.step()
        step += 1
        
        fires_extinguished = state['fires_extinguished']
        
        # Simple reward calculation
        total_reward += fires_extinguished * 10 - step * 0.1
        
        if state['terminated'] or state['truncated']:
            break
    
    return {
        'scenario_id': scenario_id,
        'total_reward': total_reward,
        'fires_extinguished': fires_extinguished,
        'steps_taken': step,
        'success': state['terminated']
    }


if __name__ == "__main__":
    compare_drones(num_scenarios=5, grid_size=25, num_fires=6)
