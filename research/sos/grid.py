"""滑移网格环境（规格见 ``research/sos/prereg-e0-e1.md``）。

**为什么是滑移的**：确定性网格能被世界模型零误差拟合，于是「预测误差的减少」恒为 0，
P2 就没有任何信息。滑移让「期望下一位置」本身带不可约误差，误差的大小才可能随覆盖度变化。
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

__all__ = [
    "ACTIONS",
    "DEFAULT_SIZE",
    "DEFAULT_SLIP",
    "DEFAULT_WALL_PROB",
    "Grid",
    "bfs_length",
    "bfs_path",
    "brute_force_length",
    "make_grid",
    "neighbours",
    "step",
]

#: (dx, dy)：上、右、下、左。x 向右、y 向上，(0, 0) 在左下角。
ACTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
DEFAULT_SIZE = 8
DEFAULT_WALL_PROB = 0.20
DEFAULT_SLIP = 0.10


@dataclass(frozen=True)
class Grid:
    """一张网格。``walls`` 是 ``(x, y)`` 的集合。"""

    size: int
    walls: frozenset[tuple[int, int]]
    start: tuple[int, int]
    goal: tuple[int, int]

    def inside(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < self.size and 0 <= y < self.size

    def blocked(self, cell: tuple[int, int]) -> bool:
        return (not self.inside(cell)) or cell in self.walls


def make_grid(
    seed: int,
    *,
    size: int = DEFAULT_SIZE,
    wall_prob: float = DEFAULT_WALL_PROB,
) -> Grid:
    """生成一张网格。**不做可解性筛除**——可解率是 E0-3 要量的东西，不能在这里被消掉。"""
    rng = random.Random(seed)
    start = (0, 0)
    goal = (size - 1, size - 1)
    cells = [(x, y) for x in range(size) for y in range(size) if (x, y) not in (start, goal)]
    n_walls = round(wall_prob * size * size)
    walls = frozenset(rng.sample(cells, min(n_walls, len(cells))))
    return Grid(size=size, walls=walls, start=start, goal=goal)


def neighbours(grid: Grid, cell: tuple[int, int]):
    for dx, dy in ACTIONS:
        candidate = (cell[0] + dx, cell[1] + dy)
        if not grid.blocked(candidate):
            yield candidate


def bfs_path(grid: Grid) -> list[tuple[int, int]] | None:
    """回到终点的最短路（含起点与终点）；不可达返回 ``None``。"""
    if grid.blocked(grid.start) or grid.blocked(grid.goal):
        return None
    parent: dict[tuple[int, int], tuple[int, int] | None] = {grid.start: None}
    queue = deque([grid.start])
    while queue:
        cell = queue.popleft()
        if cell == grid.goal:
            path = [cell]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return list(reversed(path))
        for nxt in neighbours(grid, cell):
            if nxt not in parent:
                parent[nxt] = cell
                queue.append(nxt)
    return None


def bfs_length(grid: Grid) -> int | None:
    path = bfs_path(grid)
    return None if path is None else len(path) - 1


def brute_force_length(grid: Grid) -> int | None:
    """穷举全部简单路径取最短——只在小网格上用，作为 BFS 的对照。"""
    best: list[int] = []

    def walk(cell, seen):
        if cell == grid.goal:
            best.append(len(seen) - 1)
            return
        if best and len(seen) - 1 >= min(best):
            return
        for nxt in neighbours(grid, cell):
            if nxt not in seen:
                walk(nxt, seen | {nxt})

    walk(grid.start, {grid.start})
    return min(best) if best else None


def step(
    grid: Grid,
    cell: tuple[int, int],
    action: int,
    rng: random.Random,
    *,
    p_slip: float = DEFAULT_SLIP,
) -> tuple[int, int]:
    """走一步。滑移时在其余三个动作里均匀选一个；撞墙原地不动。"""
    if rng.random() < p_slip:
        action = rng.choice([other for other in range(len(ACTIONS)) if other != action])
    dx, dy = ACTIONS[action]
    candidate = (cell[0] + dx, cell[1] + dy)
    return cell if grid.blocked(candidate) else candidate


def make_maze(seed: int, *, size: int = 11, loop: float = 0.05) -> Grid:
    """DFS 迷宫（走廊在奇数坐标，偶数坐标是墙），再打通 ``loop`` 比例的剩余内墙。

    由构造保证连通：DFS 生成的是生成树，任何两个走廊格之间都有路。
    ``size`` 必须是奇数。
    """
    if size % 2 == 0:
        raise ValueError(f"迷宫边长必须是奇数，收到 {size}。")
    rng = random.Random(seed)
    walls = {(x, y) for x in range(size) for y in range(size)}
    start = (1, 1)
    stack = [start]
    walls.discard(start)
    while stack:
        x, y = stack[-1]
        candidates = []
        for dx, dy in ((0, 2), (2, 0), (0, -2), (-2, 0)):
            nx, ny = x + dx, y + dy
            if 1 <= nx < size - 1 and 1 <= ny < size - 1 and (nx, ny) in walls:
                candidates.append((nx, ny, dx // 2, dy // 2))
        if not candidates:
            stack.pop()
            continue
        nx, ny, hx, hy = rng.choice(candidates)
        walls.discard((nx, ny))
        walls.discard((x + hx, y + hy))
        stack.append((nx, ny))
    interior = [(x, y) for x in range(1, size - 1) for y in range(1, size - 1) if (x, y) in walls]
    for cell in interior:
        if rng.random() < loop:
            walls.discard(cell)
    return Grid(size=size, walls=frozenset(walls), start=start, goal=(size - 2, size - 2))
