"""E0–E1（v2：迷宫任务，判据见 prereg-e0-e1-v2.md）。

跑法::

    python -m research.sos.run_e0_e1_v2 --out <仓外>
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from random import Random

import numpy as np

from .grid import DEFAULT_SLIP, Grid, bfs_length, brute_force_length, make_grid, make_maze, step
from .run_e0_e1 import (
    N_FOLDS,
    SEEDS,
    features,
    fit_ridge,
    greedy_policy,
    make_follow_policy,
    random_policy,
    spearman,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "sweep_results" / "sos_e0_e1_v2"

SIZES = (9, 11, 13)
GRIDS_PER_SIZE = 200
E1_SIZE = 11
TRAIN_GRIDS = 160
HOLDOUT_GRIDS = 40

#: 判据（改这里必须改预注册）。
E0_DISTINCT_MIN = 8
E0_SHORT_MAX = 14
E0_LONG_MIN = 40
E0_FOLLOW_MIN = 0.90
E0_GAP_MIN = 0.40

E1_HIT_MIN = 0.85
E1_SPEARMAN_MIN = 0.30
E1_COVERAGE_SEEDS_MIN = 4
E1_CV_MIN = 0.10


def episode_cap(grid: Grid) -> int:
    """`T = 3·L + 20`（预注册 §任务）。"""
    length = bfs_length(grid)
    return 3 * length + 20 if length is not None else 60


def run_policy(grid: Grid, policy, rng: Random) -> bool:
    cell = grid.start
    for _ in range(episode_cap(grid)):
        cell = step(grid, cell, policy(grid, cell), rng, p_slip=DEFAULT_SLIP)
        if cell == grid.goal:
            return True
    return False


def collect(grids: list[Grid], seed: int) -> tuple[np.ndarray, ...]:
    rng = Random(seed)
    states, actions, targets, grid_ids = [], [], [], []
    for grid_id, grid in enumerate(grids):
        size = grid.size
        cell = grid.start
        for _ in range(episode_cap(grid)):
            action = random_policy(grid, cell)
            nxt = step(grid, cell, action, rng, p_slip=DEFAULT_SLIP)
            states.append(cell[1] * size + cell[0])
            actions.append(action)
            targets.append((nxt[0] / (size - 1), nxt[1] / (size - 1)))
            grid_ids.append(grid_id)
            cell = nxt
    return (
        np.asarray(states, dtype=np.int64),
        np.asarray(actions, dtype=np.int64),
        np.asarray(targets, dtype=np.float64),
        np.asarray(grid_ids, dtype=np.int64),
    )


def evaluate_seed(seed: int, grids: list[Grid]) -> dict:
    size = grids[0].size
    states, actions, targets, grid_ids = collect(grids, seed)
    holdout_ids = set(range(TRAIN_GRIDS, TRAIN_GRIDS + HOLDOUT_GRIDS))
    train = ~np.isin(grid_ids, list(holdout_ids))
    holdout = ~train

    folds = np.array_split(np.flatnonzero(train), N_FOLDS)
    design = features(states, actions, size)
    predictions = [design[holdout] @ fit_ridge(design[fold], targets[fold]) for fold in folds]
    stack = np.stack(predictions)
    mean = stack.mean(axis=0)
    variance = stack.std(axis=0).mean(axis=1)

    truth = targets[holdout]
    error = np.abs(mean - truth).mean(axis=1)
    predicted_cell = np.rint(mean * (size - 1)).astype(np.int64)
    true_cell = np.rint(truth * (size - 1)).astype(np.int64)
    hit = float(np.mean(np.all(predicted_cell == true_cell, axis=1)))

    counts = np.zeros(size * size * 4, dtype=np.int64)
    keys, key_counts = np.unique(states[train] * 4 + actions[train], return_counts=True)
    counts[keys] = key_counts
    holdout_counts = counts[states[holdout] * 4 + actions[holdout]]
    low = holdout_counts <= np.median(holdout_counts)
    coverage_ok = float(variance[low].mean()) > float(variance[~low].mean())

    return {
        "seed": seed,
        "holdout": int(holdout.sum()),
        "hit": hit,
        "spearman": spearman(variance.tolist(), error.tolist()),
        "coverage_ok": coverage_ok,
        "variance_cv": float(variance.std() / variance.mean()) if variance.mean() > 0 else 0.0,
    }


def e0_checks() -> dict:
    first, second = make_maze(7), make_maze(7)
    deterministic = first.walls == second.walls and first.start == second.start

    mismatches, checked = 0, 0
    for size in (5, 7):
        for seed in range(20):
            grid = make_grid(seed, size=size)
            checked += 1
            if bfs_length(grid) != brute_force_length(grid):
                mismatches += 1

    mazes: dict[int, list[Grid]] = {}
    lengths: dict[int, list[int]] = {}
    for size in SIZES:
        grids = [make_maze(seed, size=size) for seed in range(GRIDS_PER_SIZE)]
        mazes[size] = grids
        lengths[size] = [bfs_length(grid) for grid in grids]
    solvable_all = all(all(length is not None for length in row) for row in lengths.values())
    combined = [length for row in lengths.values() for length in row if length is not None]

    rates = {}
    for name in ("random", "greedy", "follow"):
        wins, total = 0, 0
        index = 0
        for size in SIZES:
            for grid in mazes[size]:
                policy = (
                    greedy_policy
                    if name == "greedy"
                    else random_policy
                    if name == "random"
                    else None
                )
                if name == "follow":
                    policy = make_follow_policy(grid)
                    if policy is None:
                        continue
                for seed in SEEDS:
                    total += 1
                    wins += run_policy(grid, policy, Random(seed * 100_003 + index))
                index += 1
        rates[name] = wins / total if total else float("nan")

    return {
        "e0_1_deterministic": deterministic,
        "e0_2_bfs_checked": checked,
        "e0_2_bfs_mismatches": mismatches,
        "e0_3_solvable_all": solvable_all,
        "e0_4_distinct": len(set(combined)),
        "e0_4_min": min(combined),
        "e0_4_max": max(combined),
        "e0_4_per_size": {
            str(size): {"min": min(row), "max": max(row), "distinct": len(set(row))}
            for size, row in lengths.items()
        },
        "e0_5_rates": rates,
        "pass": (
            deterministic
            and mismatches == 0
            and solvable_all
            and len(set(combined)) >= E0_DISTINCT_MIN
            and min(combined) <= E0_SHORT_MAX
            and max(combined) >= E0_LONG_MIN
            and rates["follow"] >= E0_FOLLOW_MIN
            and rates["follow"] - rates["random"] >= E0_GAP_MIN
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="脉冲 SoS 的 E0–E1（v2 迷宫）")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="e0_e1_v2")
    args = parser.parse_args(argv)

    e0 = e0_checks()
    print(
        f"[E0] 确定性 {e0['e0_1_deterministic']} | BFS 对照 {e0['e0_2_bfs_checked']} 张"
        f"（不一致 {e0['e0_2_bfs_mismatches']}） | 全部可解 {e0['e0_3_solvable_all']}"
    )
    print(
        f"[E0] 难度跨度 {e0['e0_4_distinct']} 档 [{e0['e0_4_min']}, {e0['e0_4_max']}] | "
        f"分尺寸 {e0['e0_4_per_size']}"
    )
    rates = e0["e0_5_rates"]
    print(
        f"[E0] 基线：随机 {rates['random']:.3f} / 贪心 {rates['greedy']:.3f} / "
        f"跟随 {rates['follow']:.3f} | E0 {'过' if e0['pass'] else '不过'}"
    )

    result = {"kind": "spiking sos e0-e1 v2 (maze)", "e0": e0, "e1": None, "p1_ready": False}
    if e0["pass"]:
        grids = [make_maze(seed, size=E1_SIZE) for seed in range(TRAIN_GRIDS + HOLDOUT_GRIDS)]
        rows = [evaluate_seed(seed, grids) for seed in SEEDS]
        hits = statistics.median(row["hit"] for row in rows)
        rhos = statistics.median(row["spearman"] for row in rows)
        cvs = statistics.median(row["variance_cv"] for row in rows)
        coverage = sum(1 for row in rows if row["coverage_ok"])
        passed = (
            hits >= E1_HIT_MIN
            and rhos >= E1_SPEARMAN_MIN
            and coverage >= E1_COVERAGE_SEEDS_MIN
            and cvs >= E1_CV_MIN
        )
        result["e1"] = {
            "rows": rows,
            "hit_median": hits,
            "spearman_median": rhos,
            "variance_cv_median": cvs,
            "coverage_seeds": coverage,
            "pass": passed,
        }
        result["p1_ready"] = passed
        print(
            f"[E1] 命中率中位 {hits:.3f} | ρ 中位 {rhos:.3f} | 方差 CV 中位 {cvs:.3f} | "
            f"覆盖度方向 {coverage}/5 | P1 {'过' if passed else '不过'}"
        )
    else:
        print("[E1] 跳过：E0 未过。")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
