#!/usr/bin/env python3
"""
PyroGuard AI - Advanced Reinforcement Learning System
Sophisticated RL training with strategic fire prioritization and multi-objective rewards
"""

from __future__ import annotations

import os
import math
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# Import your existing wildfire simulator (no drone logic inside it)
from wildfire_env import WildfireEnvironment


# =============================================================================
#                               GYM WRAPPER ENV
# =============================================================================

class WildfireDroneEnv(gym.Env):
    """
    Gym wrapper around WildfireEnvironment for RL.
    Single- (default) and multi-agent (optional) action encoding via MultiDiscrete.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        grid_size: int = 25,
        max_steps: int = 500,
        reward_config: Optional[Dict] = None,
        enable_multi_agent: bool = False,
        num_drones: int = 1,
    ):
        super().__init__()

        # ----------- configuration
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.enable_multi_agent = enable_multi_agent
        self.num_drones = max(1, int(num_drones))
        self.current_step = 0

        # ----------- reward config
        self.reward_config = {
            "fire_extinguished": 100.0,
            "fire_spread_prevented": 50.0,
            "fire_spread_penalty": -25.0,
            "idle_penalty": -1.0,
            "strategic_priority_bonus": 200.0,
            "resource_efficiency_bonus": 25.0,
            "area_coverage_bonus": 15.0,
            "cooperation_bonus": 30.0,  # only used for multi-agent mode
            "early_intervention_bonus": 75.0,
        }
        if reward_config:
            self.reward_config.update(reward_config)

        # ----------- base sim (no internal drone)
        self.env = WildfireEnvironment(
            grid_size=grid_size,
            fire_spread_prob=0.15,
            wind_strength=0.10,
            fire_persistence=6,
            new_fire_rate=0.02,
            enable_drones=False,
        )

        # ----------- drone states (RL-controlled)
        self.drone_positions: List[Tuple[int, int]] = [(1, 1) for _ in range(self.num_drones)]
        self.drone_water_levels: List[int] = [20 for _ in range(self.num_drones)]
        self.drone_energy_levels: List[int] = [100 for _ in range(self.num_drones)]

        # ----------- trackers for rewards/metrics
        self.previous_fire_count = 0
        self.fires_extinguished_this_episode = 0
        self.strategic_fires_targeted = 0
        self.idle_steps = 0
        self.total_area_covered: set[Tuple[int, int]] = set()
        self.previous_known_fires: set[Tuple[int, int]] = set()

        self.episode_metrics = {
            "fires_extinguished": 0,
            "fire_spread_events": 0,
            "strategic_decisions": 0,
            "resource_efficiency": 0.0,
            "area_coverage": 0,
        }

        # ----------- spaces
        self._setup_spaces()

    # -------------------------------------------------------------------------
    # Render passthrough (fix for demo)
    # -------------------------------------------------------------------------

    # def render(self, mode="human"):
    #     """Forward render call to the underlying WildfireEnvironment."""
    #     if hasattr(self.env, "render"):
    #         return self.env.render(mode=mode)
    #     else:
    #         raise NotImplementedError("Underlying WildfireEnvironment has no render() method")

    # def render(self, mode="human"):
    # # """Forward render to WildfireEnvironment and overlay RL drones."""
    #     if hasattr(self.env, "render"):
    #         self.env.render(mode=mode)

    #         # Draw RL drones on top
    #         if hasattr(self.env, "screen") and self.env.screen:
    #             import pygame
    #             for i, (r, c) in enumerate(self.drone_positions):
    #                 x = c * self.env.cell_size
    #                 y = r * self.env.cell_size

    #                 # Draw blue circle for drone
    #                 pygame.draw.circle(
    #                     self.env.screen,
    #                     (0, 0, 255),
    #                     (x + self.env.cell_size // 2, y + self.env.cell_size // 2),
    #                     self.env.cell_size // 3,
    #                 )

    #                 # Drone ID text
    #                 font = pygame.font.SysFont("Arial", 12)
    #                 label = font.render(f"D{i+1}", True, (255, 255, 255))
    #                 self.env.screen.blit(label, (x + 2, y + 2))

    #             pygame.display.flip()
    #     else:
    #         raise NotImplementedError("Underlying WildfireEnvironment has no render() method")

    def render(self, mode="human"):
    # """Forward render to WildfireEnvironment and overlay RL drones (bigger & clearer)."""
        if hasattr(self.env, "render"):
            self.env.render(mode=mode)

            # Draw RL drones on top
            if hasattr(self.env, "screen") and self.env.screen:
                import pygame
                for i, (r, c) in enumerate(self.drone_positions):
                    x = c * self.env.cell_size
                    y = r * self.env.cell_size

                    # Bigger blue circle for drone
                    pygame.draw.circle(
                        self.env.screen,
                        (0, 100, 255),  # brighter blue
                        (x + self.env.cell_size // 2, y + self.env.cell_size // 2),
                        self.env.cell_size // 2,  # bigger radius
                    )

                    # Emoji or text label 🚁
                    font = pygame.font.SysFont("Arial", 20, bold=True)
                    label = font.render("🚁", True, (255, 255, 255))
                    self.env.screen.blit(label, (x + self.env.cell_size // 3, y + self.env.cell_size // 3))

                    # Optional: Drone ID (D1, D2...)
                    id_label = font.render(f"D{i+1}", True, (255, 255, 0))
                    self.env.screen.blit(id_label, (x, y))

                pygame.display.flip()
        else:
            raise NotImplementedError("Underlying WildfireEnvironment has no render() method")


    def close(self):
        """Forward close call to the underlying WildfireEnvironment."""
        if hasattr(self.env, "close"):
            self.env.close()


    # -------------------------------------------------------------------------
    # Spaces
    # -------------------------------------------------------------------------

    def _setup_spaces(self):
        # Action space: [move_dir (0..8), suppress(0/1), priority (0..8)]
        unit = [9, 2, 9]
        if self.enable_multi_agent:
            self.action_space = spaces.MultiDiscrete(unit * self.num_drones)
        else:
            self.action_space = spaces.MultiDiscrete(unit)

        # Observation space length:
        # grid + fire_age + (drone_state x drones) + env_stats(10) + strategic(50)
        obs_len = (
            self.grid_size * self.grid_size
            + self.grid_size * self.grid_size
            + self.num_drones * 4
            + 10
            + 50
        )

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_len,), dtype=np.float32
        )

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.env.reset()
        self.current_step = 0

        self.drone_positions = [(1, 1) for _ in range(self.num_drones)]
        self.drone_water_levels = [20 for _ in range(self.num_drones)]
        self.drone_energy_levels = [100 for _ in range(self.num_drones)]

        self.previous_fire_count = 0
        self.fires_extinguished_this_episode = 0
        self.strategic_fires_targeted = 0
        self.idle_steps = 0
        self.total_area_covered.clear()
        self.previous_known_fires.clear()

        self.episode_metrics = {
            k: (0.0 if k in ("resource_efficiency",) else 0)
            for k in self.episode_metrics.keys()
        }

        # spawn some initial fires to learn on
        self.env.spawn_random_fires(np.random.randint(2, 5))

        return self._get_observation(), self._get_info()

    # -------------------------------------------------------------------------
    # Step
    # -------------------------------------------------------------------------

    def step(self, action):
        self.current_step += 1

        actions_taken = self._process_rl_actions(action)

        # progress environment
        env_state = self.env.step()

        # reward
        reward = self._calculate_strategic_reward(actions_taken, env_state)

        # termination / truncation
        terminated = self._check_termination(env_state)
        truncated = self.current_step >= self.max_steps

        obs = self._get_observation()
        info = self._get_info()
        info.update(actions_taken)

        return obs, reward, terminated, truncated, info

    # -------------------------------------------------------------------------
    # Action helpers
    # -------------------------------------------------------------------------

    def _process_rl_actions(self, action) -> Dict:
        """
        Handles either single-agent or multi-agent MultiDiscrete actions.
        """
        if self.enable_multi_agent:
            # reshape sequence into per-drone triplets
            action = np.array(action).reshape(self.num_drones, 3)
            out = {"movements": [], "suppressions": [], "strategic_targets": [], "resource_usages": []}
            for i in range(self.num_drones):
                res = self._process_single_drone_action(i, action[i])
                out["movements"].append(res["movement"])
                out["suppressions"].append(res["suppression"])
                out["strategic_targets"].append(res["strategic_target"])
                out["resource_usages"].append(res["resource_usage"])
            return out

        # single agent
        res = self._process_single_drone_action(0, action)
        return res

    # def _process_single_drone_action(self, drone_id: int, action) -> Dict:
    #     action = np.array(action).flatten().astype(int)
    #     move_dir = int(action[0])
    #     suppress = bool(int(action[1]))
    #     priority = int(action[2])

    #     # movement deltas
    #     dir_map = {
    #         0: (0, 0),  1: (-1, 0), 2: (-1, 1), 3: (0, 1),  4: (1, 1),
    #         5: (1, 0),  6: (1, -1), 7: (0, -1), 8: (-1, -1)
    #     }
    #     dr, dc = dir_map.get(move_dir, (0, 0))

    #     cur_r, cur_c = self.drone_positions[drone_id]
    #     water = self.drone_water_levels[drone_id]
    #     energy = self.drone_energy_levels[drone_id]

    #     result = {"movement": None, "suppression": None, "strategic_target": None, "resource_usage": 0}

    #     # generic energy tick
    #     self.drone_energy_levels[drone_id] = max(0, energy - 1)

    #     # move (clip to bounds)
    #     nr = int(np.clip(cur_r + dr, 0, self.grid_size - 1))
    #     nc = int(np.clip(cur_c + dc, 0, self.grid_size - 1))
    #     if (nr, nc) != (cur_r, cur_c):
    #         self.drone_positions[drone_id] = (nr, nc)
    #         self.total_area_covered.add((nr, nc))
    #         result["movement"] = (nr, nc)

    #     # try suppression
    #     if suppress and water > 0:
    #         sup = self._attempt_fire_suppression(drone_id, priority)
    #         if sup["success"]:
    #             self.drone_water_levels[drone_id] = max(0, water - 2)
    #             self.fires_extinguished_this_episode += 1
    #             result["suppression"] = sup
    #             result["resource_usage"] = 2

    #     # refuel at base (near (1,1))
    #     if water < 5 or energy < 20:
    #         if abs(nr - 1) + abs(nc - 1) <= 1:
    #             self.drone_water_levels[drone_id] = 20
    #             self.drone_energy_levels[drone_id] = 100

    #     return result

    def _process_single_drone_action(self, drone_id: int, action) -> Dict:
        action = np.array(action).flatten().astype(int)
        move_dir = int(action[0])
        suppress = bool(int(action[1]))
        priority = int(action[2])

        # movement deltas
        dir_map = {
            0: (0, 0),  1: (-1, 0), 2: (-1, 1), 3: (0, 1),  4: (1, 1),
            5: (1, 0),  6: (1, -1), 7: (0, -1), 8: (-1, -1)
        }
        dr, dc = dir_map.get(move_dir, (0, 0))

        # ✅ Speed boost (move more than 1 cell per step)
        speed = 2   # increase this number to make drone fly faster
        dr *= speed
        dc *= speed

        cur_r, cur_c = self.drone_positions[drone_id]
        water = self.drone_water_levels[drone_id]
        energy = self.drone_energy_levels[drone_id]

        result = {"movement": None, "suppression": None, "strategic_target": None, "resource_usage": 0}

        # generic energy tick (costs slightly more energy if moving fast)
        self.drone_energy_levels[drone_id] = max(0, energy - (1 * speed))

        # move (clip to bounds)
        nr = int(np.clip(cur_r + dr, 0, self.grid_size - 1))
        nc = int(np.clip(cur_c + dc, 0, self.grid_size - 1))
        if (nr, nc) != (cur_r, cur_c):
            self.drone_positions[drone_id] = (nr, nc)
            self.total_area_covered.add((nr, nc))
            result["movement"] = (nr, nc)

        # try suppression
        if suppress and water > 0:
            sup = self._attempt_fire_suppression(drone_id, priority)
            if sup["success"]:
                self.drone_water_levels[drone_id] = max(0, water - 2)
                self.fires_extinguished_this_episode += 1
                result["suppression"] = sup
                result["resource_usage"] = 2

        # refuel at base (near (1,1))
        if water < 5 or energy < 20:
            if abs(nr - 1) + abs(nc - 1) <= 1:
                self.drone_water_levels[drone_id] = 20
                self.drone_energy_levels[drone_id] = 100

        return result


    def _attempt_fire_suppression(self, drone_id: int, priority: int) -> Dict:
        """Suppress any fire within 1-cell Chebyshev range; pick via strategic rule."""
        r, c = self.drone_positions[drone_id]
        grid = self.env.grid

        nearby = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.grid_size and 0 <= cc < self.grid_size:
                    if grid[rr, cc] in (2, 5):  # FIRE or FIRE_INTENSE
                        nearby.append((rr, cc))

        if not nearby:
            return {"success": False, "reason": "no_fires_in_range"}

        target = self._select_strategic_target(nearby, priority)
        if target is None:
            return {"success": False, "reason": "target_selection_failed"}

        # perform suppression
        self.env.grid[target[0], target[1]] = 3  # BURNED
        self.env.fire_age[target[0], target[1]] = 0

        sv = self._calculate_strategic_value(target)
        if sv > 0.7:
            self.strategic_fires_targeted += 1

        return {"success": True, "position": target, "strategic_value": float(sv)}

    def _select_strategic_target(self, fires: List[Tuple[int, int]], priority: int) -> Optional[Tuple[int, int]]:
        """Pick a target by priority: best, top-3 random, or any random."""
        if not fires:
            return None

        scored = [(f, self._calculate_strategic_value(f)) for f in fires]
        scored.sort(key=lambda x: x[1], reverse=True)

        if priority <= 2:
            return scored[0][0]
        elif priority <= 5:
            top = scored[: min(3, len(scored))]
            return random.choice([f for f, _ in top])
        else:
            return random.choice(fires)

    # -------------------------------------------------------------------------
    # Strategic scoring and reward engineering
    # -------------------------------------------------------------------------

    def _calculate_strategic_value(self, pos: Tuple[int, int]) -> float:
        """Heuristic scoring for a fire's importance."""
        r, c = pos
        grid = self.env.grid
        score = 0.0

        # flammable neighbors
        flammable = 0.0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.grid_size and 0 <= cc < self.grid_size:
                    ct = grid[rr, cc]
                    if ct in (1, 6, 7, 9):  # TREE, DENSE, SPARSE, GRASS
                        flammable += 1.0
                        if ct == 6:  # DENSE_FOREST
                            flammable += 0.5

        score += flammable * 0.15

        # intensity
        if grid[r, c] == 5:  # FIRE_INTENSE
            score += 0.3

        # edge distance (further from edge = slightly more critical)
        edge_dist = min(r, c, self.grid_size - 1 - r, self.grid_size - 1 - c)
        if edge_dist > 2:
            score += 0.1

        # cluster size
        cluster = self._count_fire_cluster_size(pos)
        if cluster > 3:
            score += 0.2

        return float(min(1.0, score))

    def _count_fire_cluster_size(self, start: Tuple[int, int]) -> int:
        """BFS cluster size around a fire cell (capped for safety)."""
        grid = self.env.grid
        if grid[start[0], start[1]] not in (2, 5):
            return 0

        q = [start]
        visited = {start}
        size = 0

        while q and size < 50:
            r, c = q.pop()
            if grid[r, c] not in (2, 5):
                continue
            size += 1
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.grid_size and 0 <= cc < self.grid_size:
                        p = (rr, cc)
                        if p not in visited and grid[rr, cc] in (2, 5):
                            visited.add(p)
                            q.append(p)

        return size

    def _calculate_strategic_reward(self, actions: Dict, env_state: Dict) -> float:
        """Multi-term reward with strategic bonuses/penalties."""
        reward = 0.0
        current_fires = int(env_state["fires_active"])

        # extinguishment (delta)
        extinguished = max(0, self.previous_fire_count - current_fires)
        if extinguished > 0:
            reward += extinguished * self.reward_config["fire_extinguished"]
            self.episode_metrics["fires_extinguished"] += extinguished

        # spread penalty
        if current_fires > self.previous_fire_count:
            spread = current_fires - self.previous_fire_count
            reward += spread * self.reward_config["fire_spread_penalty"]
            self.episode_metrics["fire_spread_events"] += spread

        # strategic bonus (single-agent form)
        if isinstance(actions.get("suppression"), dict) and actions["suppression"]:
            sv = float(actions["suppression"].get("strategic_value", 0.0))
            if sv > 0.7:
                reward += self.reward_config["strategic_priority_bonus"]
                self.episode_metrics["strategic_decisions"] += 1

        # resource efficiency
        usage = 0
        if "resource_usage" in actions:
            usage = int(actions["resource_usage"])
        elif "resource_usages" in actions:
            usage = int(sum(int(x or 0) for x in actions["resource_usages"]))

        if usage > 0:
            eff = extinguished / max(1, usage)
            reward += eff * self.reward_config["resource_efficiency_bonus"]
            self.episode_metrics["resource_efficiency"] += eff

        # coverage bonus
        covered = len(self.total_area_covered)
        if covered > self.episode_metrics["area_coverage"]:
            inc = covered - self.episode_metrics["area_coverage"]
            reward += inc * self.reward_config["area_coverage_bonus"]
            self.episode_metrics["area_coverage"] = covered

        # idle penalty (single-agent logic: no movement & no suppression)
        idle = False
        if "movement" in actions and "suppression" in actions:
            idle = (actions["movement"] is None) and (actions["suppression"] is None)
        if idle:
            self.idle_steps += 1
            if self.idle_steps > 3:
                reward += self.reward_config["idle_penalty"]
        else:
            self.idle_steps = 0

        # track fires set (for early intervention heuristic)
        self.previous_fire_count = current_fires
        current_set = set()
        fr = np.where((self.env.grid == 2) | (self.env.grid == 5))
        for r, c in zip(fr[0], fr[1]):
            current_set.add((int(r), int(c)))

        # early intervention — if new fires appeared but count stayed low next step
        if len(current_set - self.previous_known_fires) > 0:
            # weak heuristic (kept simple)
            reward += self.reward_config["early_intervention_bonus"] * 0.0  # optional: disable by default

        self.previous_known_fires = current_set
        return float(reward)

    # -------------------------------------------------------------------------
    # Observations / Infos
    # -------------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        parts: List[np.ndarray] = []

        # 1) grid (normalize cell ids by 10)
        grid_norm = (self.env.grid.flatten().astype(np.float32)) / 10.0
        parts.append(grid_norm)

        # 2) fire age
        fire_age = np.clip(self.env.fire_age.flatten().astype(np.float32) / 10.0, 0.0, 1.0)
        parts.append(fire_age)

        # 3) drones
        for i in range(self.num_drones):
            r, c = self.drone_positions[i]
            drone_state = np.array(
                [r / self.grid_size, c / self.grid_size, self.drone_water_levels[i] / 20.0, self.drone_energy_levels[i] / 100.0],
                dtype=np.float32,
            )
            parts.append(drone_state)

        # 4) env stats (10)
        current_fire_count = int(np.sum((self.env.grid == 2) | (self.env.grid == 5)))
        env_stats = np.array(
            [
                self.env.step_count / 1000.0,
                current_fire_count / float(self.grid_size * self.grid_size),
                self.env.wind_direction / 360.0,
                float(self.env.wind_strength),
                len(self.total_area_covered) / float(self.grid_size * self.grid_size),
                self.fires_extinguished_this_episode / 50.0,
                self.strategic_fires_targeted / 20.0,
                self.episode_metrics["strategic_decisions"] / 10.0,
                float(self.episode_metrics["resource_efficiency"]),
                self.idle_steps / 10.0,
            ],
            dtype=np.float32,
        )
        parts.append(env_stats)

        # 5) strategic analysis (50)
        parts.append(self._get_strategic_fire_analysis())

        obs = np.concatenate(parts).astype(np.float32)

        # pad/trim to expected length just in case
        expected = self.observation_space.shape[0]
        if obs.shape[0] < expected:
            obs = np.concatenate([obs, np.zeros(expected - obs.shape[0], dtype=np.float32)])
        elif obs.shape[0] > expected:
            obs = obs[:expected]
        return obs

    def _get_strategic_fire_analysis(self) -> np.ndarray:
        out = np.zeros(50, dtype=np.float32)

        fr = np.where((self.env.grid == 2) | (self.env.grid == 5))
        fires = list(zip(fr[0], fr[1]))
        if not fires:
            return out

        # up to 10 fires; for each: [value, dist2drone, intensity, cluster, edge]
        for i, (r, c) in enumerate(fires[:10]):
            base = i * 5
            sv = self._calculate_strategic_value((int(r), int(c)))
            # distance to nearest drone
            mind = min(abs(r - dr) + abs(c - dc) for (dr, dc) in self.drone_positions)
            intensity = 1.0 if self.env.grid[r, c] == 5 else 0.5
            cluster = self._count_fire_cluster_size((int(r), int(c))) / 20.0
            edge = min(r, c, self.grid_size - 1 - r, self.grid_size - 1 - c) / float(self.grid_size)

            out[base : base + 5] = [sv, mind / self.grid_size, intensity, cluster, edge]

        return out

    def _get_info(self) -> Dict:
        return {
            "episode_metrics": dict(self.episode_metrics),
            "fires_active": int(np.sum((self.env.grid == 2) | (self.env.grid == 5))),
            "total_vegetation": int(
                np.sum((self.env.grid == 1) | (self.env.grid == 6) | (self.env.grid == 7) | (self.env.grid == 9))
            ),
            "step": int(self.current_step),
            "drone_positions": list(self.drone_positions),
            "drone_resources": {
                "water_levels": list(self.drone_water_levels),
                "energy_levels": list(self.drone_energy_levels),
            },
        }

    # -------------------------------------------------------------------------
    # Termination
    # -------------------------------------------------------------------------

    def _check_termination(self, env_state: Dict) -> bool:
        if env_state["fires_active"] == 0 and env_state["total_vegetation"] == 0:
            return True

        depleted = all(
            (w <= 0 and e <= 0) for (w, e) in zip(self.drone_water_levels, self.drone_energy_levels)
        )
        return depleted


# =============================================================================
#                               CALLBACK
# =============================================================================

class AdvancedTrainingCallback(BaseCallback):
    """
    Saves best model by episode reward (moving window) and periodic checkpoints.
    """

    def __init__(self, save_freq: int = 20000, save_path: str = "./models/", verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.best_mean_reward = -float("inf")

    def _on_step(self) -> bool:
        # Checkpointing
        if self.n_calls % self.save_freq == 0:
            p = os.path.join(self.save_path, f"model_{self.n_calls}")
            self.model.save(p)
            if self.verbose:
                print(f"[Callback] Saved checkpoint at {p}")

        # Track rolling ep rewards (if any)
        if len(self.model.ep_info_buffer) > 0:
            rs = [e["r"] for e in self.model.ep_info_buffer]
            mean_r = float(np.mean(rs))
            if mean_r > self.best_mean_reward:
                self.best_mean_reward = mean_r
                self.model.save(os.path.join(self.save_path, "best_model"))
                if self.verbose:
                    print(f"[Callback] New best mean reward: {mean_r:.2f} (saved best_model)")
        return True


# =============================================================================
#                               TRAIN / EVAL / DEPLOY
# =============================================================================

def _make_train_env(grid_size: int, reward_cfg: Optional[Dict] = None):
    def thunk():
        env = WildfireDroneEnv(
            grid_size=grid_size,
            max_steps=400,
            reward_config=reward_cfg,
            enable_multi_agent=False,
            num_drones=1,
        )
        return Monitor(env)
    return thunk


def train_advanced_rl_agent(
    total_timesteps: int = 200_000,
    algorithm: str = "ppo",
    grid_size: int = 25,
    save_path: str = "./models/",
    device: str = "cpu",
):
    print("Starting PyroGuard Advanced RL Training")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Device: {device}")
    print("=" * 60)

    reward_cfg = {
        "fire_extinguished": 150.0,
        "fire_spread_prevented": 75.0,
        "fire_spread_penalty": -30.0,
        "idle_penalty": -2.0,
        "strategic_priority_bonus": 300.0,
        "resource_efficiency_bonus": 40.0,
        "area_coverage_bonus": 20.0,
        "early_intervention_bonus": 100.0,
    }

    # vectorized envs (2 envs keeps CPU happy)
    vec_env = make_vec_env(_make_train_env(grid_size, reward_cfg), n_envs=2)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    # model
    algorithm = algorithm.lower().strip()
    if algorithm == "ppo":
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=5e-4,
            n_steps=1024,
            batch_size=32,
            n_epochs=8,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                activation_fn=nn.ReLU,
            ),
            device=device,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
        )
    elif algorithm == "dqn":
        model = DQN(
            "MlpPolicy",
            vec_env,
            learning_rate=2e-4,
            buffer_size=50_000,
            learning_starts=10_000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=5_000,
            exploration_fraction=0.15,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            policy_kwargs=dict(net_arch=[128, 128], activation_fn=nn.ReLU),
            device=device,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    cb = AdvancedTrainingCallback(save_freq=20_000, save_path=save_path, verbose=1)

    start = datetime.now()
    model.learn(
        total_timesteps=int(total_timesteps),
        callback=cb,
        log_interval=10,
        tb_log_name=f"pyroguard_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        progress_bar=False,
    )
    duration = datetime.now() - start

    # final saves
    final_model_path = os.path.join(save_path, f"final_model_{algorithm}")
    model.save(final_model_path)
    vec_env.save(os.path.join(save_path, "vec_normalize.pkl"))

    print("\nTraining complete")
    print(f"Time: {duration}")
    print(f"Saved: {final_model_path}")
    return model, vec_env


def evaluate_trained_model(
    model_path: str,
    algorithm: str = "ppo",   # 👈 added
    vec_env_path: Optional[str] = None,
    num_episodes: int = 5,
    render: bool = False,
    grid_size: int = 25,
):
    print(f"Evaluating model: {model_path}")

    eval_env = WildfireDroneEnv(
        grid_size=grid_size,
        max_steps=800,
        enable_multi_agent=False,
        num_drones=1,
    )

    # VecNormalize (disable reward normalization during eval)
    if vec_env_path and os.path.exists(vec_env_path):
        vec = DummyVecEnv([lambda: Monitor(eval_env)])
        vec = VecNormalize.load(vec_env_path, vec)
        vec.training = False
        vec.norm_reward = False
    else:
        vec = DummyVecEnv([lambda: Monitor(eval_env)])

    # Load model explicitly
    if algorithm.lower() == "ppo":
        model = PPO.load(model_path)
    elif algorithm.lower() == "dqn":
        model = DQN.load(model_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    ep_returns = []
    ep_metrics = []

    for ep in range(1, num_episodes + 1):
        obs = vec.reset()
        done = [False]
        ep_ret = 0.0
        steps = 0

        print(f"\nEpisode {ep}/{num_episodes}")
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec.step(action)
            ep_ret += float(reward[0])
            steps += 1

            if render and hasattr(eval_env, "render"):
                eval_env.render()

            if steps % 200 == 0:
                fires = info[0].get("fires_active", None)
                print(f"  Step {steps}: reward {float(reward[0]):.2f} | fires {fires}")

        ep_returns.append(ep_ret)
        ep_metrics.append(info[0].get("episode_metrics", {}))
        print(f"Episode done: return={ep_ret:.2f}, steps={steps}")

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Avg return: {np.mean(ep_returns):.2f} ± {np.std(ep_returns):.2f}")
    print(f"Best: {np.max(ep_returns):.2f} | Worst: {np.min(ep_returns):.2f}")

    if ep_metrics:
        keys = ep_metrics[0].keys()
        print("\nAvg metrics:")
        for k in keys:
            vals = [float(m.get(k, 0)) for m in ep_metrics]
            print(f"  {k}: {np.mean(vals):.2f}")

    return ep_returns, ep_metrics



# def create_inference_agent(model_path: str, vec_env_path: Optional[str] = None):
#     """
#     Build a lightweight inference wrapper. (Uses the SB3 model internally;
#     simple and robust for deployment; you can later export to TorchScript/ONNX.)
#     """
#     print("Preparing inference agent…")

#     # detect algo
#     algo = "ppo" if "ppo" in os.path.basename(model_path).lower() else "dqn"
#     model = PPO.load(model_path, device="cpu") if algo == "ppo" else DQN.load(model_path, device="cpu")

#     # optional VecNormalize params
#     vec_norm = None
#     if vec_env_path and os.path.exists(vec_env_path):
#         # we don't need a live env here; only stats
#         dummy_env = DummyVecEnv([lambda: WildfireDroneEnv()])
#         vec_norm = VecNormalize.load(vec_env_path, dummy_env)
#         vec_norm.training = False
#         vec_norm.norm_reward = False

#     class InferenceAgent:
#         def __init__(self, model, vec_norm=None):
#             self.model = model
#             self.vec_norm = vec_norm

#         def predict(self, obs: np.ndarray, deterministic: bool = True):
#             if self.vec_norm is not None:
#                 obs = self.vec_norm.normalize_obs(obs)
#             action, _ = self.model.predict(obs, deterministic=deterministic)
#             return action

#         def save_optimized(self, path: str):
#             # save SB3 model + (optional) vec stats alongside
#             self.model.save(path)
#             if self.vec_norm is not None:
#                 self.vec_norm.save(os.path.splitext(path)[0] + "_vecnorm.pkl")
#             print(f"Saved optimized deployment package to: {path}")

#     return InferenceAgent(model, vec_norm)

def create_inference_agent(model_path: str, vec_env_path: Optional[str] = None):
    """
    Build a lightweight inference wrapper. (Uses the SB3 PPO model internally;
    simple and robust for deployment; you can later export to TorchScript/ONNX.)
    """
    print("Preparing inference agent…")

    # Always PPO (since our training uses PPO)
    model = PPO.load(model_path, device="cpu")

    # optional VecNormalize params
    vec_norm = None
    if vec_env_path and os.path.exists(vec_env_path):
        # we don't need a live env here; only stats
        dummy_env = DummyVecEnv([lambda: WildfireDroneEnv()])
        vec_norm = VecNormalize.load(vec_env_path, dummy_env)
        vec_norm.training = False
        vec_norm.norm_reward = False

    class InferenceAgent:
        def __init__(self, model, vec_norm=None):
            self.model = model
            self.vec_norm = vec_norm

        def predict(self, obs: np.ndarray, deterministic: bool = True):
            if self.vec_norm is not None:
                obs = self.vec_norm.normalize_obs(obs)
            action, _ = self.model.predict(obs, deterministic=deterministic)
            return action

        def save_optimized(self, path: str):
            # save SB3 model + (optional) vec stats alongside
            self.model.save(path)
            if self.vec_norm is not None:
                self.vec_norm.save(os.path.splitext(path)[0] + "_vecnorm.pkl")
            print(f"Saved optimized deployment package to: {path}")

    return InferenceAgent(model, vec_norm)



def run_training_pipeline(
    total_timesteps: int = 200_000,
    algorithm: str = "ppo",
    grid_size: int = 25,
    evaluate: bool = True,
    create_deployment_model: bool = True,
):
    print("PyroGuard AI - Advanced RL Training Pipeline")
    print("=" * 60)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.abspath(f"./models/pyroguard_{algorithm}_{ts}")
    os.makedirs(save_dir, exist_ok=True)

    # Train
    print("\nStage 1: Training")
    print("-" * 40)
    model, vec_env = train_advanced_rl_agent(
        total_timesteps=total_timesteps,
        algorithm=algorithm,
        grid_size=grid_size,
        save_path=save_dir,
        device="cpu",
    )

    # Evaluate
    if evaluate:
        print("\nStage 2: Evaluation")
        print("-" * 40)
        best = os.path.join(save_dir, "best_model.zip")
        final = os.path.join(save_dir, f"final_model_{algorithm}.zip")
        vec_path = os.path.join(save_dir, "vec_normalize.pkl")

        model_to_eval = best if os.path.exists(best) else final
        if not os.path.exists(model_to_eval):
            # fallback (SB3 adds .zip automatically)
            model_to_eval = model_to_eval[:-4]

        evaluate_trained_model(
            model_path=model_to_eval,
            vec_env_path=vec_path,
            num_episodes=5,
            render=False,
            grid_size=grid_size,
        )

    # Build deployment package
    if create_deployment_model:
        print("\nStage 3: Deployment Package")
        print("-" * 40)
        best = os.path.join(save_dir, "best_model.zip")
        if not os.path.exists(best):
            best = os.path.join(save_dir, f"final_model_{algorithm}.zip")

        agent = create_inference_agent(best, os.path.join(save_dir, "vec_normalize.pkl"))
        agent.save_optimized(os.path.join(save_dir, "deployment_model.zip"))

    print("\nPipeline complete.")
    print(f"Artifacts in: {save_dir}")
    print(
        "Saved files:\n"
        "  - best_model.zip (if best mean reward found)\n"
        f"  - final_model_{algorithm}.zip\n"
        "  - vec_normalize.pkl\n"
        "  - deployment_model.zip\n"
    )
    return save_dir


# =============================================================================
#                               CLI
# =============================================================================

def main():
    import argparse

    p = argparse.ArgumentParser(description="PyroGuard Advanced RL")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--algorithm", type=str, default="ppo", choices=["ppo", "dqn"])
    p.add_argument("--grid-size", type=int, default=25)
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--deploy", action="store_true")
    p.add_argument("--eval-only", type=str, default=None, help="Path to model (.zip) to evaluate only")
    args = p.parse_args()

    if args.eval_only:
        # eval-only mode
        vec_path_guess = os.path.join(os.path.dirname(args.eval_only), "vec_normalize.pkl")
        evaluate_trained_model(
            model_path=args.eval_only,
            vec_env_path=vec_path_guess if os.path.exists(vec_path_guess) else None,
            num_episodes=5,
            render=False,
            grid_size=args.grid_size,
        )
        return

    print("Training Configuration:")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Algorithm: {args.algorithm.upper()}")
    print("  Device: CPU")
    print()

    run_training_pipeline(
        total_timesteps=args.timesteps,
        algorithm=args.algorithm,
        grid_size=args.grid_size,
        evaluate=args.evaluate,
        create_deployment_model=args.deploy,
    )

    print("\nTo evaluate later, run:")
    print(f"python {os.path.basename(__file__)} --eval-only <path_to_model>.zip")


# =============================================================================
#                           Quick sanity training
# =============================================================================

def quick_train_and_test():
    """Very quick 50K PPO run (smaller grid) to verify everything wires up."""
    print("Quick training (50k) starting…")
    path = run_training_pipeline(
        total_timesteps=50_000,
        algorithm="ppo",
        grid_size=20,
        evaluate=True,
        create_deployment_model=True,
    )
    print(f"Quick run done. Artifacts: {path}")
    return path


if __name__ == "__main__":
    main()

