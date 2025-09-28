import gym
from gym import spaces
import numpy as np
import pygame
import random

# Cell states
EMPTY = 0       # Tree
BURNING = 1     # Fire
BURNED = 2      # Ash
WATER = 3       # River / puddles
GRASS = 4       # Grassland (spreads slower)

class WildfireGymEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, grid_size=25, render_mode="human"):
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0, high=4, shape=(grid_size, grid_size), dtype=np.int8
        )
        self.action_space = spaces.Discrete(1)  # no drone yet

        # Pygame setup
        self._cell_px = 25
        self._screen = None
        self._clock = None

        self.grid = None
        self.running = True
        self.fire_started = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Add river
        river_col = self.grid_size // 2
        self.grid[:, river_col] = WATER

        # Random puddles
        for _ in range(self.grid_size // 3):
            i, j = np.random.randint(0, self.grid_size, 2)
            self.grid[i, j] = WATER

        # Grassland patches (bigger clusters)
        for _ in range(3):
            gx, gy = np.random.randint(0, self.grid_size, 2)
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        if self.grid[nx, ny] == EMPTY:
                            self.grid[nx, ny] = GRASS

        self.running = True
        self.fire_started = False
        return self.grid.copy(), {}

    def step(self, action=None):
        if not self.fire_started:
            # Do nothing until fire is ignited
            return self.grid.copy(), 0, False, False, {}

        new_grid = self.grid.copy()
        spread_happened = False

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == BURNING:
                    new_grid[i, j] = BURNED
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = i+dx, j+dy
                        if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                            if self.grid[ni, nj] == EMPTY:
                                if random.random() < 0.9:  # strong spread
                                    new_grid[ni, nj] = BURNING
                                    spread_happened = True
                            elif self.grid[ni, nj] == GRASS:
                                if random.random() < 0.5:  # slower spread
                                    new_grid[ni, nj] = BURNING
                                    spread_happened = True

        self.grid = new_grid

        # Keep running until everything that can burn is burned
        terminated = not np.any(self.grid == BURNING) and self.fire_started
        truncated = False
        if terminated:
            self.running = False

        return self.grid.copy(), 0, terminated, truncated, {}

    def render(self):
        if self._screen is None:
            pygame.init()
            w = self.grid_size * self._cell_px
            self._screen = pygame.display.set_mode((w, w))
            pygame.display.set_caption("Wildfire Gym")
            self._clock = pygame.time.Clock()

        surface = pygame.Surface((self.grid_size*self._cell_px,
                                  self.grid_size*self._cell_px))
        colors = {
            EMPTY:  (34, 139, 34),   # green trees
            BURNING:(255, 0, 0),     # fire red
            BURNED: (70, 70, 70),    # ash gray
            WATER:  (0, 0, 255),     # blue
            GRASS:  (144, 238, 144)  # light green grassland
        }

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rect = pygame.Rect(j*self._cell_px, i*self._cell_px,
                                   self._cell_px, self._cell_px)
                pygame.draw.rect(surface, colors[self.grid[i,j]], rect)
                pygame.draw.rect(surface, (0,0,0), rect, 1)

        self._screen.blit(surface, (0,0))
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

        # Handle quit + mouse fire ignition
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                i, j = y // self._cell_px, x // self._cell_px
                if self.grid[i, j] in [EMPTY, GRASS]:
                    self.grid[i, j] = BURNING
                    self.fire_started = True

    def close(self):
        if self._screen is not None:
            pygame.display.quit()
            pygame.quit()
            self._screen = None
            self._clock = None
