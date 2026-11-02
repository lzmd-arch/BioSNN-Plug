"""滑移迷宫 fixture 的守卫（见 `benchmarks/maze/README.md`）。

迷宫完全由 `(seed, size)` 决定，不存数据文件；这里把几个最短路长度钉死，
任何改动 `make_maze` 的行为都会被立刻抓到。
"""

from __future__ import annotations

import random

from research.sos.grid import (
    DEFAULT_SLIP,
    bfs_length,
    brute_force_length,
    make_grid,
    make_maze,
    step,
)

#: 钉住的 (size, seed) → 最短路长度。改生成算法就必须同时改这里，并说明为什么。
PINNED = {
    (9, 0): 28,
    (9, 1): 16,
    (9, 2): 28,
    (11, 0): 36,
    (11, 1): 16,
    (11, 2): 16,
    (13, 0): 40,
    (13, 1): 30,
    (13, 2): 52,
}


def test_pinned_shortest_paths() -> None:
    for (size, seed), expected in PINNED.items():
        assert bfs_length(make_maze(seed, size=size)) == expected


def test_maze_is_deterministic() -> None:
    assert make_maze(7, size=11).walls == make_maze(7, size=11).walls


def test_every_maze_is_solvable() -> None:
    for size in (9, 11, 13):
        for seed in range(20):
            assert bfs_length(make_maze(seed, size=size)) is not None


def test_bfs_matches_bruteforce_on_small_grids() -> None:
    for size in (5, 7):
        for seed in range(5):
            grid = make_grid(seed, size=size)
            assert bfs_length(grid) == brute_force_length(grid)


def test_walls_stop_movement() -> None:
    grid = make_maze(0, size=9)
    rng = random.Random(0)
    cell = grid.start
    # 朝一个确定是墙的方向走，位置不变（贴墙走一次就够）
    for action in range(4):
        dx, dy = ((0, 1), (1, 0), (0, -1), (-1, 0))[action]
        if grid.blocked((cell[0] + dx, cell[1] + dy)):
            assert step(grid, cell, action, rng, p_slip=0.0) == cell
            break
    else:  # pragma: no cover - 起点四面都通的情况
        assert DEFAULT_SLIP >= 0
