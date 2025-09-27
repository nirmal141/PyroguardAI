# Simple wildfire cellular automaton utilities

from __future__ import annotations
import numpy as np
from typing import Tuple

# Cell states
TREE, BURNING, BURNED, DRONE_EXTINGUISHED, COMPLETELY_BURNED = 0, 1, 2, 3, 4

def neighbors4(x: int, y: int, w: int, h: int):
    if x > 0:         yield (x-1, y)
    if x < w-1:       yield (x+1, y)
    if y > 0:         yield (x, y-1)
    if y < h-1:       yield (x, y+1)

def ignite_random(grid: np.ndarray, k: int = 1, rng: np.random.Generator | None = None):
    """Set k random TREE cells to BURNING."""
    rng = rng or np.random.default_rng()
    h, w = grid.shape
    tree_positions = np.argwhere(grid == TREE)
    if len(tree_positions) == 0:
        return
    k = min(k, len(tree_positions))
    idx = rng.choice(len(tree_positions), size=k, replace=False)
    for (y, x) in tree_positions[idx]:
        grid[y, x] = BURNING

def step_fire(grid: np.ndarray, p_spread: float, p_burnout: float,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """
    One simulation step of the wildfire process.
    - Burning -> Burned with p_burnout
    - Burned -> Completely Burned after some time
    - Spread to 4-neighbors TREE with p_spread
    Returns the next grid (does not modify input).
    """
    rng = rng or np.random.default_rng()
    h, w = grid.shape
    new_grid = grid.copy()

    # burnouts: BURNING -> BURNED (extinguished by fire)
    burning_mask = (grid == BURNING)
    burnout_draw = rng.random(size=grid.shape)
    new_grid[(burning_mask) & (burnout_draw < p_burnout)] = BURNED

    # burned progression: BURNED -> COMPLETELY_BURNED (over time)
    burned_mask = (grid == BURNED)
    burned_draw = rng.random(size=grid.shape)
    new_grid[(burned_mask) & (burned_draw < 0.1)] = COMPLETELY_BURNED  # 10% chance to become completely burnt

    # spread (loop over burning cells for clarity; still fast for hackathon sizes)
    burning_indices = np.argwhere(grid == BURNING)
    for y, x in burning_indices:
        for nx, ny in neighbors4(x, y, w, h):
            if grid[ny, nx] == TREE and rng.random() < p_spread:
                new_grid[ny, nx] = BURNING

    return new_grid

def count_states(grid: np.ndarray) -> Tuple[int, int, int, int, int]:
    """Return (#TREE, #BURNING, #BURNED, #DRONE_EXTINGUISHED, #COMPLETELY_BURNED)."""
    t = int(np.sum(grid == TREE))
    b = int(np.sum(grid == BURNING))
    d = int(np.sum(grid == BURNED))
    de = int(np.sum(grid == DRONE_EXTINGUISHED))
    cb = int(np.sum(grid == COMPLETELY_BURNED))
    return t, b, d, de, cb
