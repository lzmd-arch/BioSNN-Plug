"""E6：分辨率实验——只估 H1' 的效应量与区间，不判通过（见 prereg-e6.md）。

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
OUT = REPO / "sweep_results" / "sos_e6"

SIZES = (9, 11, 13)
TRAIN_MAZES = 100
CONFIRM_MAZES = 200
EPOCHS = 20
LEARNING_RATE = 0.05
SEEDS = tuple(range(20))

#: 判据（改这里必须改预注册）。
E5_TRAIN_MIN = 0.50
E5_H1_MIN = 0.05
E5_STEP_RATIO = 1.50


def key_of(grid, cell, heading: int) -> tuple[int, int, int, int]:
    """状态 = (相对终点位移, 四邻是否不可走 4 位, 朝向)。E5 加的是最后一项。"""
    bits = 0
    for index, (dx, dy) in enumerate(ACTIONS):
        if grid.blocked((cell[0] + dx, cell[1] + dy)):
            bits |= 1 << index
    return (grid.goal[0] - cell[0], grid.goal[1] - cell[1], bits, heading)


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


def train(arm: str, grids, seed: int) -> dict:
    """朝向取自**原始轨迹**里前一步的动作（预注册 §朝向怎么算）。"""
    rng = Random(seed)
    theta: dict = {}
    for _ in range(EPOCHS):
        for index, grid in enumerate(grids):
            base = opt_trace(grid)
            if not base:
                continue
            source = base if arm == "opt" else dfs_trace(grid, seed * 100_003 + index)
            if not source:
                continue
            if len(source) <= len(base):
                chosen = list(range(len(source)))
            else:
                chosen = rng.sample(range(len(source)), len(base))
            for position in chosen:
                cell, action = source[position]
                heading = source[position - 1][1] if position > 0 else 4
                key = key_of(grid, cell, heading)
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
        heading = 4
        for index in range(mission(grid)):
            key = key_of(grid, cell, heading)
            action = int(np.argmax(theta.get(key, np.zeros(len(ACTIONS)))))
            cell = step(grid, cell, action, rng, p_slip=DEFAULT_SLIP)
            heading = action
            if cell == grid.goal:
                wins += 1
                steps.append(index + 1)
                break
    return {"success": wins / len(grids), "steps": statistics.fmean(steps) if steps else 0.0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="脉冲 SoS 的 E6（效应量估计）")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="e6")
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
    # 配对差：每个种子一个 d_i（预注册 §本实验不设通过/不通过）
    deltas = [
        s["success"] - o["success"]
        for o, s in zip(opt["rows"]["confirm"], search["rows"]["confirm"], strict=True)
    ]
    median = statistics.median(deltas)
    rng = Random(7)
    boot = []
    for _ in range(10_000):
        sample = [deltas[rng.randrange(len(deltas))] for _ in range(len(deltas))]
        boot.append(statistics.median(sample))
    boot.sort()
    low, high = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot)) - 1]
    if low > 0 and low > 0.05:
        reading = "H1' 成立（区间下界 > 0.05）"
    elif low > 0:
        reading = "有差别但未达 0.05 的绝对门槛"
    else:
        reading = "这个分辨率下检测不到差别"
    out = {
        "kind": "spiking sos e6 (resolution: effect size only)",
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "train_mazes": TRAIN_MAZES,
        "confirm_mazes": CONFIRM_MAZES,
        "arms": results,
        "paired_deltas": deltas,
        "delta_median": median,
        "delta_ci95": [low, high],
        "reading": reading,
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
    print(
        f"[E6] search−opt 配对差中位 {median:+.3f}，95% 区间 [{low:+.3f}, {high:+.3f}] "
        f"（n={len(deltas)} 个种子）→ {reading}"
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
