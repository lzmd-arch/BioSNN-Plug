"""E2：串行 SoS 的三臂比较（判据见 prereg-e2.md）。

策略表示：以「相对终点的位移」为状态（裁剪到 [-3,3]²），4 个动作的 softmax。
三个臂只差 reward：A = −surprisal、B = 朝终点近了 +1、C = 到达终点 +1。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from random import Random

import numpy as np

from .grid import ACTIONS, DEFAULT_SLIP, Grid, bfs_length, make_maze, step

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "sweep_results" / "sos_e2"

SIZES = (9, 11, 13)
TRAIN_MAZES = 100
CONFIRM_MAZES = 100
EPOCHS = 20
LEARNING_RATE = 0.05
GAMMA = 0.99
ALPHA = 1.0
SEEDS = (0, 1, 2, 3, 4)
ARMS = ("A", "B", "C")

E2_ENTROPY_RATIO = 0.5
E2_C_VS_B = 0.5
E2_AC_RATIO = 0.5


def key_of(grid: Grid, cell: tuple[int, int]) -> tuple[int, int]:
    return (
        max(-3, min(3, grid.goal[0] - cell[0])),
        max(-3, min(3, grid.goal[1] - cell[1])),
    )


def softmax(theta: np.ndarray) -> np.ndarray:
    shifted = theta - theta.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def support(grid: Grid, cell: tuple[int, int]) -> list[tuple[int, int]]:
    cells = [cell]
    for dx, dy in ACTIONS:
        candidate = (cell[0] + dx, cell[1] + dy)
        if grid.inside(candidate):
            cells.append(candidate)
    return cells


def surprisal(grid: Grid, table: dict, cell, action, nxt) -> float:
    counter = table.get((cell, action))
    cells = support(grid, cell)
    total = sum(counter.values()) if counter else 0
    hits = counter[nxt] if counter else 0
    return -math.log((hits + ALPHA) / (total + ALPHA * len(cells)))


def mission(grid: Grid) -> int:
    length = bfs_length(grid)
    return 3 * length + 20 if length is not None else 60


def episode(grid: Grid, theta: dict, arm: str, rng: Random) -> tuple[list, list, bool, list]:
    cell = grid.start
    table: dict = {}
    trajectory, rewards, visited = [], [], []
    goal = grid.goal
    done = False
    for _ in range(mission(grid)):
        key = key_of(grid, cell)
        probs = softmax(theta.setdefault(key, np.zeros(len(ACTIONS))))
        action = int(rng.choices(range(len(ACTIONS)), weights=probs)[0])
        nxt = step(grid, cell, action, rng, p_slip=DEFAULT_SLIP)
        if arm == "A":
            reward = -surprisal(grid, table, cell, action, nxt)
            table.setdefault((cell, action), Counter())[nxt] += 1
        elif arm == "B":
            before = abs(goal[0] - cell[0]) + abs(goal[1] - cell[1])
            after = abs(goal[0] - nxt[0]) + abs(goal[1] - nxt[1])
            reward = 1.0 if after < before else -1.0
        else:
            reward = 0.0
        trajectory.append((key, action))
        rewards.append(reward)
        visited.append(nxt)
        cell = nxt
        if cell == goal:
            done = True
            if arm == "C":
                rewards[-1] = 1.0
            break
    return trajectory, rewards, done, visited


def update(theta: dict, baseline: dict, trajectory, rewards) -> None:
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + GAMMA * running
        returns.append(running)
    returns.reverse()
    for (key, action), value in zip(trajectory, returns, strict=True):
        probs = softmax(theta.setdefault(key, np.zeros(len(ACTIONS))))
        advantage = value - baseline.setdefault(key, 0.0)
        gradient = -probs
        gradient[action] += 1.0
        theta[key] = theta[key] + LEARNING_RATE * advantage * gradient
        baseline[key] = 0.95 * baseline[key] + 0.05 * value


def train(arm: str, grids: list[Grid], seed: int) -> dict:
    rng = Random(seed)
    theta: dict = {}
    baseline: dict = {}
    for _ in range(EPOCHS):
        for grid in grids:
            trajectory, rewards, _, _ = episode(grid, theta, arm, rng)
            update(theta, baseline, trajectory, rewards)
    return theta


def evaluate(theta: dict, grids: list[Grid], seed: int) -> dict:
    rng = Random(seed + 50_000)
    wins, steps, visited = 0, [], Counter()
    for grid in grids:
        cell = grid.start
        for index in range(mission(grid)):
            key = key_of(grid, cell)
            action = int(np.argmax(theta.get(key, np.zeros(len(ACTIONS)))))
            cell = step(grid, cell, action, rng, p_slip=DEFAULT_SLIP)
            visited[cell] += 1
            if cell == grid.goal:
                wins += 1
                steps.append(index + 1)
                break
    total = sum(visited.values())
    entropy = -sum((c / total) * math.log(c / total) for c in visited.values()) if total else 0.0
    return {
        "success": wins / len(grids),
        "steps": statistics.fmean(steps) if steps else 0.0,
        "entropy": entropy,
    }


def mazes(offset: int, count: int) -> list[Grid]:
    return [make_maze(offset + index, size=SIZES[index % len(SIZES)]) for index in range(count)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="脉冲 SoS 的 E2（三臂）")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="e2")
    args = parser.parse_args(argv)

    train_grids = mazes(0, TRAIN_MAZES)
    confirm_grids = mazes(1000, CONFIRM_MAZES)

    results: dict[str, dict] = {}
    for arm in ARMS:
        train_rows, confirm_rows = [], []
        for seed in SEEDS:
            theta = train(arm, train_grids, seed)
            train_rows.append(evaluate(theta, train_grids, seed))
            confirm_rows.append(evaluate(theta, confirm_grids, seed))
        results[arm] = {
            "train": {
                "success": statistics.median(row["success"] for row in train_rows),
                "entropy": statistics.median(row["entropy"] for row in train_rows),
            },
            "confirm": {
                "success": statistics.median(row["success"] for row in confirm_rows),
                "steps": statistics.median(row["steps"] for row in confirm_rows),
                "entropy": statistics.median(row["entropy"] for row in confirm_rows),
            },
            "rows": {"train": train_rows, "confirm": confirm_rows},
        }

    a, b, c = (results[arm]["confirm"] for arm in ARMS)
    train_gap = results["A"]["train"]["success"] - results["C"]["train"]["success"]
    criteria = {
        "e2_0_not_degenerate": a["entropy"] >= E2_ENTROPY_RATIO * c["entropy"],
        "e2_1_sparse_learns": c["success"] >= E2_C_VS_B * b["success"],
        "e2_2_process_beats_outcome": a["success"] >= c["success"] + E2_AC_RATIO * train_gap,
        "train_gap": train_gap,
    }

    out = {
        "kind": "spiking sos e2 (three arms)",
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "train_mazes": TRAIN_MAZES,
        "confirm_mazes": CONFIRM_MAZES,
        "arms": results,
        "criteria": criteria,
        "h1_supported": bool(
            criteria["e2_0_not_degenerate"] and criteria["e2_2_process_beats_outcome"]
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )

    for arm in ARMS:
        row = results[arm]["confirm"]
        print(
            f"[{arm}] confirm 成功率 {row['success']:.3f} | 平均步数 {row['steps']:.1f} | "
            f"访问熵 {row['entropy']:.3f}"
        )
    print(
        f"[E2-0] A 熵 {a['entropy']:.3f} vs 0.5×C {0.5 * c['entropy']:.3f} → "
        f"{'不退化' if criteria['e2_0_not_degenerate'] else '退化，H1 不可测'}"
    )
    print(
        f"[E2-1] C {c['success']:.3f} vs 0.5×B {0.5 * b['success']:.3f} → "
        f"{'稀疏臂可用' if criteria['e2_1_sparse_learns'] else '稀疏臂太弱'}"
    )
    print(
        f"[E2-2] A {a['success']:.3f} vs C {c['success']:.3f} + 0.5×训练差 {0.5 * train_gap:+.3f} → "
        f"{'H1 成立' if criteria['e2_2_process_beats_outcome'] else 'H1 不成立'}"
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
