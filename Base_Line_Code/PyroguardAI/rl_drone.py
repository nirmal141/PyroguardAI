#!/usr/bin/env python3
"""
RL Drone for PyroGuard
- Wraps WildfireEnvironment into a Gymnasium env
- Trains a PPO agent that moves and extinguishes fires
Run training:   python rl_drone.py --train
Run inference:  python rl_drone.py --play models/ppo_drone.zip
"""

from __future__ import annotations
import os
import math
import argparse
from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from stable_baselines3 import PPO  # you can switch to DQN if you prefer
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
except Exception as e:
    PPO = None  # allow import without SB3 for now

# Import your teammate's environment
from wildfire_env import WildfireEnvironment  # <-- already provided in your repo


# --------- Constants mirror wildfire_env for readability ----------
EMPTY = WildfireEnvironment.EMPTY
TREE = WildfireEnvironment.TREE
FIRE = WildfireEnvironment.FIRE
BURNED = WildfireEnvironment.BURNED
WATER = WildfireEnvironment.WATER
FIRE_INTENSE = WildfireEnvironment.FIRE_INTENSE
DENSE_FOREST = WildfireEnvironment.DENSE_FOREST
SPARSE_FOREST = WildfireEnvironment.SPARSE_FOREST
ROCK = WildfireEnvironment.ROCK
GRASS = WildfireEnvironment.GRASS

FLAMMABLE = {TREE, DENSE_FOREST, SPARSE_FOREST, GRASS}
FIRE_SET = {FIRE, FIRE_INTENSE}


# -------------------- Config --------------------
@dataclass
class RLDroneConfig:
    grid_size: int = 30
    patch: int = 9                 # square local observation size (odd number)
    max_steps: int = 600
    water_capacity: int = 20
    energy_capacity: int = 400
    detection_range: int = 5       # for extinguish targeting (within 1 step action we still use a range=1)
    extinguish_range: int = 1
    move_cost: float = 0.0         # optional small movement penalty
    step_idle_penalty: float = -0.05
    reward_extinguish: float = 10.0
    reward_fire_delta_pos: float = 2.0
    reward_fire_delta_neg: float = -1.0
    wasted_water_penalty: float = -1.0
    seed: int | None = None


# -------------------- Gym Wrapper --------------------
class WildfireDroneEnv(gym.Env):
    """
    Single-drone RL environment wrapped around WildfireEnvironment.
    Action space: Discrete(6) -> 0:stay, 1:up, 2:down, 3:left, 4:right, 5:extinguish
    Observation: flattened (patch x patch) local grid (normalized) + [row, col, water, energy]
    """
    metadata = {"render_modes": ["human"], "render_fps": 8}

    def __init__(self, cfg: RLDroneConfig):
        super().__init__()
        self.cfg = cfg

        # Underlying sim WITHOUT built-in rule-based drone
        self.sim = WildfireEnvironment(
            grid_size=cfg.grid_size,
            fire_spread_prob=0.12,   # slower & learnable
            initial_tree_density=0.75,
            wind_strength=0.08,
            fire_persistence=6,
            new_fire_rate=0.02,
            enable_drones=False
        )

        # Drone state
        self.row = 1
        self.col = 1
        self.water = cfg.water_capacity
        self.energy = cfg.energy_capacity
        self.steps = 0

        # Spaces
        obs_len = cfg.patch * cfg.patch + 4
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_len,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(6)

        # Caches
        self._last_fire_count = 0
        self._rng = np.random.RandomState(cfg.seed)

    # ------------- Helpers -------------
    def _clip(self, r, c):
        return int(np.clip(r, 0, self.cfg.grid_size - 1)), int(np.clip(c, 0, self.cfg.grid_size - 1))

    def _reset_drone(self):
        # spawn near (1,1) but not on water/rock
        for _ in range(100):
            r = self._rng.randint(1, min(4, self.cfg.grid_size - 2))
            c = self._rng.randint(1, min(4, self.cfg.grid_size - 2))
            if self.sim.grid[r, c] not in (WATER, ROCK):
                self.row, self.col = r, c
                break
        self.water = self.cfg.water_capacity
        self.energy = self.cfg.energy_capacity
        self.steps = 0

    def _local_patch(self):
        g = self.sim.grid
        k = self.cfg.patch // 2
        r0, c0 = self.row, self.col
        patch = np.zeros((self.cfg.patch, self.cfg.patch), dtype=np.float32)
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                rr, cc = r0 + i, c0 + j
                if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                    patch[i + k, j + k] = g[rr, cc]
                else:
                    patch[i + k, j + k] = ROCK  # out of bounds as rock (non-traversable)
        # normalize to [-1,1]
        patch = (patch / 9.0) * 2.0 - 1.0
        return patch

    def _obs(self):
        patch = self._local_patch().flatten()
        extras = np.array([
            self.row / (self.cfg.grid_size - 1) * 2 - 1,
            self.col / (self.cfg.grid_size - 1) * 2 - 1,
            self.water / self.cfg.water_capacity * 2 - 1,
            self.energy / self.cfg.energy_capacity * 2 - 1
        ], dtype=np.float32)
        return np.concatenate([patch, extras]).astype(np.float32)

    def _count_fire(self):
        g = self.sim.grid
        return int(np.sum((g == FIRE) | (g == FIRE_INTENSE)))

    # ------------- Gym API -------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self._reset_drone()
        self._last_fire_count = self._count_fire()
        return self._obs(), {}

    def step(self, action: int):
        self.steps += 1
        self.energy = max(0, self.energy - 1)

        # Apply action BEFORE sim spread, so extinguish affects this step
        reward = 0.0
        info = {}
        g = self.sim.grid

        # Movement
        dr = dc = 0
        if action == 1: dr = -1
        elif action == 2: dr = +1
        elif action == 3: dc = -1
        elif action == 4: dc = +1

        if action in (1, 2, 3, 4):
            nr, nc = self._clip(self.row + dr, self.col + dc)
            # disallow moving into water/rock
            if g[nr, nc] not in (WATER, ROCK):
                self.row, self.col = nr, nc
                reward += self.cfg.move_cost

        # Extinguish (adjacent von Neumann within range 1)
        extinguished = 0
        water_used = 0
        if action == 5 and self.water > 0:
            candidates = []
            for drr, dcc in [(-1,0),(1,0),(0,-1),(0,1)]:
                rr, cc = self._clip(self.row + drr, self.col + dcc)
                if g[rr, cc] in FIRE_SET:
                    candidates.append((rr, cc))
            # If standing on fire (danger), extinguish self cell first
            if g[self.row, self.col] in FIRE_SET:
                candidates.insert(0, (self.row, self.col))

            if candidates:
                target = candidates[0]
                g[target[0], target[1]] = BURNED
                self.sim.fire_age[target[0], target[1]] = 0
                self.water -= 1
                water_used += 1
                extinguished += 1
                reward += self.cfg.reward_extinguish
            else:
                # sprayed water but no fire nearby (if we still count it as use)
                self.water -= 1
                water_used += 1

        # Let fire dynamics evolve
        # Note: we temporarily disable the built-in drone processing already (enable_drones=False)
        state = self.sim.step()

        # Fire delta reward
        fire_now = self._count_fire()
        delta = fire_now - self._last_fire_count
        if delta < 0:
            reward += self.cfg.reward_fire_delta_pos * (-delta)
        elif delta > 0:
            reward += self.cfg.reward_fire_delta_neg * (delta)
        self._last_fire_count = fire_now

        # # Wasted water penalty
        # if water_used > 0 and extinguished == 0:
        #     reward += self.cfg.wasted_water_penalty * water_used

        # Step idle penalty (always)
        reward += self.cfg.step_idle_penalty

        # Done conditions
        done = False
        if self.steps >= self.cfg.max_steps:
            done = True
        if self.energy <= 0:
            done = True
        # Optional: end if no vegetation left or no fires for a while (kept simple)
        # If all fires out and no new fires appear for N steps — skipped for now.

        return self._obs(), reward, done, False, info

    # Render by delegating to underlying sim, then draw a simple drone marker
    def render(self):
        cont = self.sim.render()
        # draw drone overlay (small crosshair) on the sim screen
        try:
            import pygame
            if self.sim.screen is not None:
                cs = self.sim.cell_size
                x = self.col * cs + 20 + cs // 2
                y = self.row * cs + 20 + cs // 2
                pygame.draw.circle(self.sim.screen, (0, 200, 255), (x, y), 6, 2)
                pygame.draw.circle(self.sim.screen, (0, 200, 255), (x, y), 2)
                pygame.display.flip()
        except Exception:
            pass
        return cont

    def close(self):
        self.sim.close()


# -------------------- Training / Inference --------------------
def train_ppo(save_path: str = "models/ppo_drone.zip", total_timesteps: int = 200_000):
    cfg = RLDroneConfig()
    env = WildfireDroneEnv(cfg)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # SB3 wrappers
    def make_env():
        return Monitor(WildfireDroneEnv(cfg))

    vec_env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="runs/ppo_drone",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"✅ Saved PPO model to: {save_path}")


def play_model(model_path: str = "models/ppo_drone.zip"):
    if PPO is None:
        raise RuntimeError("Stable-Baselines3 not installed")
    cfg = RLDroneConfig()
    env = WildfireDroneEnv(cfg)
    model = PPO.load(model_path)

    obs, _ = env.reset()
    try:
        while True:
            env.render()               # you can click to ignite fires while the agent runs
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(int(action))
            if done:
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


# -------------------- CLI --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train PPO drone")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--play", type=str, default="", help="Path to trained model")
    args = parser.parse_args()

    if args.train:
        if PPO is None:
            raise RuntimeError("Stable-Baselines3 not installed: pip install stable-baselines3")
        train_ppo(total_timesteps=args.timesteps)
        return

    if args.play:
        play_model(args.play)
        return

    # default demo: quick play with random actions just to verify wiring
    print("No flags supplied. Quick smoke-test with random policy (Ctrl+C to stop).")
    env = WildfireDroneEnv(RLDroneConfig())
    obs, _ = env.reset()
    try:
        while True:
            env.render()
            action = env.action_space.sample()
            obs, r, done, _, _ = env.step(action)
            if done:
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "demo"], default="train")
    args = parser.parse_args()

    if args.mode == "train":
        train_ppo(
            save_path="models/ppo_drone.zip",
            total_timesteps=3_000_000 
        )
    else:
        main()   # your demo loop

