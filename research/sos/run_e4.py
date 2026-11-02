"""E4：轨迹监督 + 四邻墙状态（判据见 prereg-e4.md）。

与 E3 唯一的差别：状态里加了四邻的墙信息。E3 的根因是「状态里没有墙」——
同一个相对位移对应许多格子、最优动作各不相同，只观测位置的任务是 POMDP。
"""

from __future__ import annotations

import argparse
import json
import statistics
from itertools import pairwise
from pathlib import Path
from random import Random

import numpy as np

from .grid import ACTIONS, DEFAULT_SLIP, bfs_path, step
from .run_e2_v2 import mazes, mission, softmax

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "sweep_results" / "sos_e4"

SIZES = (9, 11, 13)
TRAIN_MAZES = 100
CONFIRM_MAZES = 100
EPOCHS = 20
LEARNING_RATE = 0.05
SEEDS = (0, 1, 2, 3, 4)

#: 判据（改这里必须改预注册）。
E4_TRAIN_MIN = 0.50
E4_H1_MIN = 0.05
E4_STEP_RATIO = 1.50


def key_of(grid, cell) -> tuple[int, int, int]:
    """状态 = (相对终点位移, 四邻是否不可走的 4 位)。E4 的唯一改动。"""
    bits = 0
    for index, (dx, dy) in enumerate(ACTIONS):
        if grid.blocked((cell[0] + dx, cell[1] + dy)):
            bits |= 1 << index
    return (grid.goal[0] - cell[0], grid.goal[1] - cell[1], bits)


def opt_trace(grid) -> list[tuple[tuple[int, int], int]]:
    """最短路上每一步的动作。"""
    path = bfs_path(grid)
    if path is None:
        return []
    trace = []
    for cell, nxt in pairwise(path):
        for action, (dx, dy) in enumerate(ACTIONS):
            if (cell[0] + dx, cell[1] + dy) == nxt:
                trace.append((cell, action))
                break
    return trace


def dfs_trace(grid, seed: int = 0) -> list[tuple[tuple[int, int], int]]:
    """DFS 的尝试序列：每次朝未访问的邻居走一步都记一条，回溯本身不产生样本。

    **邻居顺序按 seed 打乱**——固定顺序（上右下左）在一张完美迷宫里几乎总是先走对分支，
    于是 D_search 与 D_opt 逐条相同，两臂就没有对比了（实测中位比 1.00）。
    """
    rng = Random(seed)
    seen = {grid.start}
    stack = [grid.start]
    trace: list[tuple[tuple[int, int], int]] = []
    while stack:
        cell = stack[-1]
        if cell == grid.goal:
            break
        order = list(range(len(ACTIONS)))
        rng.shuffle(order)
        moved = False
        for action in order:
            dx, dy = ACTIONS[action]
            nxt = (cell[0] + dx, cell[1] + dy)
            if grid.blocked(nxt) or nxt in seen:
                continue
            trace.append((cell, action))
            seen.add(nxt)
            stack.append(nxt)
            moved = True
            break
        if not moved:
            stack.pop()
    return trace


def align(trace, count: int, rng: Random):
    """按预注册把样本数对齐到 ``count``：多了就均匀抽，少了就原样返回。"""
    if len(trace) <= count:
        return list(trace)
    return rng.sample(trace, count)


def train(arm: str, grids, seed: int) -> dict:
    rng = Random(seed)
    theta: dict = {}
    for _ in range(EPOCHS):
        for index, grid in enumerate(grids):
            base = opt_trace(grid)
            if not base:
                continue
            rows = (
                align(base, len(base), rng)
                if arm == "opt"
                else align(dfs_trace(grid, seed * 100_003 + index), len(base), rng)
            )
            for cell, action in rows:
                key = key_of(grid, cell)
                params = theta.setdefault(key, np.zeros(len(ACTIONS)))
                probs = softmax(params)
                gradient = -probs
                gradient[action] += 1.0
                theta[key] = params + LEARNING_RATE * gradient
    return theta


def evaluate(theta: dict, grids, seed: int) -> dict:
    rng = Random(seed + 70_000)
    wins, steps = 0, []
    for grid in grids:
        cell = grid.start
        for index in range(mission(grid)):
            key = key_of(grid, cell)
            action = int(np.argmax(theta.get(key, np.zeros(len(ACTIONS)))))
            cell = step(grid, cell, action, rng, p_slip=DEFAULT_SLIP)
            if cell == grid.goal:
                wins += 1
                steps.append(index + 1)
                break
    return {"success": wins / len(grids), "steps": statistics.fmean(steps) if steps else 0.0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="脉冲 SoS 的 E4（四邻墙状态）")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="e4")
    args = parser.parse_args(argv)

    train_grids = mazes(0, TRAIN_MAZES)
    confirm_grids = mazes(1000, CONFIRM_MAZES)

    results: dict[str, dict] = {}
    for arm in ("opt", "search"):
        train_rows, confirm_rows = [], []
        for seed in SEEDS:
            theta = train(arm, train_grids, seed)
            train_rows.append(evaluate(theta, train_grids, seed))
            confirm_rows.append(evaluate(theta, confirm_grids, seed))
        results[arm] = {
            "train": {
                "success": statistics.median(row["success"] for row in train_rows),
            },
            "confirm": {
                "success": statistics.median(row["success"] for row in confirm_rows),
                "steps": statistics.median(row["steps"] for row in confirm_rows),
            },
            "rows": {"train": train_rows, "confirm": confirm_rows},
        }

    opt, search = results["opt"], results["search"]
    gap_pilot = search["train"]["success"] - opt["train"]["success"]
    criteria = {
        "e4_0_both_learn": min(opt["train"]["success"], search["train"]["success"]) >= E4_TRAIN_MIN,
        "e4_1_process_beats_result": search["confirm"]["success"] - opt["confirm"]["success"]
        >= E4_H1_MIN,
        "e4_2_not_step_inflation": search["confirm"]["steps"]
        <= E4_STEP_RATIO * max(opt["confirm"]["steps"], 1e-9),
        "pilot_gap": gap_pilot,
    }
    out = {
        "kind": "spiking sos e4 (trace supervision + wall bits)",
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "train_mazes": TRAIN_MAZES,
        "confirm_mazes": CONFIRM_MAZES,
        "arms": results,
        "criteria": criteria,
        "h1_prime_supported": bool(
            criteria["e4_0_both_learn"]
            and criteria["e4_1_process_beats_result"]
            and criteria["e4_2_not_step_inflation"]
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )

    for arm in ("opt", "search"):
        print(
            f"[{arm}] 训练池 {results[arm]['train']['success']:.3f} | "
            f"confirm {results[arm]['confirm']['success']:.3f} | "
            f"步数 {results[arm]['confirm']['steps']:.1f}"
        )
    print(f"[E4-0] 两臂训练池 ≥{E4_TRAIN_MIN} → {'过' if criteria['e4_0_both_learn'] else '不过'}")
    print(
        f"[E4-1] H1' 绝对差 ≥{E4_H1_MIN} → {'成立' if criteria['e4_1_process_beats_result'] else '不成立'}"
    )
    print(
        f"[E4-2] 步数比 ≤{E4_STEP_RATIO} → {'过' if criteria['e4_2_not_step_inflation'] else '不过'}"
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
