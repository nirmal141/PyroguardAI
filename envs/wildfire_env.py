from __future__ import annotations
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

import os
import time

from .fire_dynamics import TREE, BURNING, BURNED, ignite_random, step_fire, count_states

class FireSimEnv(gym.Env):
    """
    2D wildfire spread environment (no drones yet).
    Obs: HxW grid of {TREE,BURNING,BURNED}
    Act: no-op (fire evolves stochastically)
    """
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(self, width=10, height=10, p_spread=0.3, p_burnout=0.1, max_steps=100, ignitions=(1,3), seed: int | None = None):
        super().__init__()
        self.width = int(width)
        self.height = int(height)
        self.p_spread = float(p_spread)
        self.p_burnout = float(p_burnout)
        self.max_steps = int(max_steps)
        self.ignitions_range = tuple(ignitions)

        # Fire-only env: single dummy action to satisfy API
        self.action_space = spaces.Discrete(1)

        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.height, self.width), dtype=np.int8
        )

        # RNGs
        self.py_rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.grid = None
        self.steps = 0

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self.py_rng = random.Random(seed)
            self.np_rng = np.random.default_rng(seed)

        self.steps = 0
        self.grid = np.full((self.height, self.width), TREE, dtype=np.int8)

        # ignite k random fires
        k_min, k_max = self.ignitions_range
        k = self.py_rng.randint(k_min, k_max)
        ignite_random(self.grid, k=k, rng=self.np_rng)

        return self.grid.copy(), {}

    def step(self, action=None):
        self.steps += 1

        # Update fire
        self.grid = step_fire(self.grid, self.p_spread, self.p_burnout, rng=self.np_rng)

        # Reward (placeholder): negative proportional to active fires
        _, burning, _ = count_states(self.grid)
        reward = -float(burning)

        terminated = (burning == 0)
        truncated = (self.steps >= self.max_steps)

        info = {"steps": self.steps, "burning": burning}
        return self.grid.copy(), reward, terminated, truncated, info


    def render(self, mode="human"):
        if not hasattr(self, "screen") and mode == "human":
            pygame.init()
            self.cell_size = 40
            self.screen = pygame.display.set_mode((self.width*self.cell_size, self.height*self.cell_size))
        
        if mode == "ansi":
            # ASCII fallback
            symbols = {0:"🌲", 1:"🔥", 2:"⬛"}
            return "\n".join("".join(symbols[int(self.grid[y,x])] for x in range(self.width)) for y in range(self.height))

        if mode == "human":
            # Clear screen with white background
            self.screen.fill((255, 255, 255))
            
            # Define symbols and colors
            symbols = {0: "🌲", 1: "🔥", 2: "⬛"}  # Tree, Fire, Burned
            colors = {0: (34, 139, 34), 1: (255, 69, 0), 2: (50, 50, 50)}  # Green, Red, Gray
            
            # Set up font for symbols
            font_size = min(self.cell_size - 4, 32)  # Make symbols fit in cells
            font = pygame.font.Font(None, font_size)
            
            for y in range(self.height):
                for x in range(self.width):
                    cell_state = int(self.grid[y, x])
                    
                    # Draw background color
                    rect = pygame.Rect(x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, colors[cell_state], rect)
                    pygame.draw.rect(self.screen, (0,0,0), rect, 1)
                    
                    # Draw symbol
                    symbol = symbols[cell_state]
                    text_surface = font.render(symbol, True, (0, 0, 0))  # Black text
                    
                    # Center the symbol in the cell
                    text_rect = text_surface.get_rect()
                    text_rect.center = (x*self.cell_size + self.cell_size//2, 
                                      y*self.cell_size + self.cell_size//2)
                    self.screen.blit(text_surface, text_rect)
            
            pygame.display.flip()


    def close(self):
        pass
