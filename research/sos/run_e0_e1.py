"""E0–E1：环境、三个无学习基线、世界模型与不确定度（判据见 prereg-e0-e1.md）。

跑法::

    python -m research.sos.run_e0_e1 --out <仓外>

E0 全过才进 E1；E1 四条全过才是 ``p1_ready = true``。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import numpy as np

from .grid import (
    ACTIONS,
    DEFAULT_SLIP,
    Grid,
    bfs_length,
    bfs_path,
    brute_force_length,
    make_grid,
    step,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "sweep_results" / "sos_e0_e1"

EPISODE_STEPS = 60
N_GRIDS = 200
TRAIN_GRIDS = 160
HOLDOUT_GRIDS = 40
N_FOLDS = 5
RIDGE_LAMBDA = 1.0
SEEDS = (0, 1, 2, 3, 4)

#: 判据（改这里必须改预注册或其 amend）。
E0_SOLVABLE_MIN = 0.95
E0_DISTINCT_MIN = 4
E0_SHORT_MAX = 6
E0_LONG_MIN = 18
E0_FOLLOW_MIN = 0.60
E0_GAP_MIN = 0.30

E1_HIT_MIN = 0.85
E1_SPEARMAN_MIN = 0.30
E1_COVERAGE_SEEDS_MIN = 4
E1_CV_MIN = 0.10


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2 + 1
        for offset in range(position, end + 1):
            ranks[order[offset]] = average
        position = end + 1
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    a, b = _ranks(left), _ranks(right)
    mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    var_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    var_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
    return 0.0 if var_a == 0 or var_b == 0 else cov / (var_a * var_b)


def run_policy(grid: Grid, policy, rng: random.Random, *, p_slip: float) -> bool:
    cell = grid.start
    for _ in range(EPISODE_STEPS):
        cell = step(grid, cell, policy(grid, cell), rng, p_slip=p_slip)
        if cell == grid.goal:
            return True
    return False


def random_policy(_grid: Grid, _cell) -> int:
    return random.randrange(len(ACTIONS))


def greedy_policy(grid: Grid, cell) -> int:
    best, best_distance = 0, None
    for action, (dx, dy) in enumerate(ACTIONS):
        candidate = (cell[0] + dx, cell[1] + dy)
        if grid.blocked(candidate):
            continue
        distance = abs(candidate[0] - grid.goal[0]) + abs(candidate[1] - grid.goal[1])
        if best_distance is None or distance < best_distance:
            best, best_distance = action, distance
    return best


def make_follow_policy(grid: Grid):
    path = bfs_path(grid)
    if path is None:
        return None
    index = {cell: position for position, cell in enumerate(path)}

    def policy(_grid: Grid, cell) -> int:
        position = index.get(cell)
        target = path[0] if position is None or position + 1 >= len(path) else path[position + 1]
        for action, (dx, dy) in enumerate(ACTIONS):
            if (cell[0] + dx, cell[1] + dy) == target:
                return action
        return 0

    return policy


def collect(grids: list[Grid], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    rows, actions, targets, grid_ids = [], [], [], []
    size = grids[0].size
    for grid_id, grid in enumerate(grids):
        cell = grid.start
        for _ in range(EPISODE_STEPS):
            action = random_policy(grid, cell)
            nxt = step(grid, cell, action, rng, p_slip=DEFAULT_SLIP)
            rows.append(cell[1] * size + cell[0])
            actions.append(action)
            targets.append((nxt[0] / (size - 1), nxt[1] / (size - 1)))
            grid_ids.append(grid_id)
            cell = nxt
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(actions, dtype=np.int64),
        np.asarray(targets, dtype=np.float64),
        np.asarray(grid_ids, dtype=np.int64),
    )


def features(states: np.ndarray, actions: np.ndarray, size: int) -> np.ndarray:
    design = np.zeros((len(states), size * size * len(ACTIONS)), dtype=np.float64)
    design[np.arange(len(states)), states * len(ACTIONS) + actions] = 1.0
    return design


def fit_ridge(design: np.ndarray, targets: np.ndarray, lam: float = RIDGE_LAMBDA) -> np.ndarray:
    gram = design.T @ design + lam * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ targets)


def evaluate_seed(seed: int, grids: list[Grid], size: int) -> dict:
    states, actions, targets, grid_ids = collect(grids, seed)
    holdout_grids = set(range(TRAIN_GRIDS, TRAIN_GRIDS + HOLDOUT_GRIDS))
    train = ~np.isin(grid_ids, list(holdout_grids))
    holdout = ~train

    # 5 个不相交子集各拟合一个模型，预测的离散度就是"方差"。
    folds = np.array_split(np.flatnonzero(train), N_FOLDS)
    design_all = features(states, actions, size)
    predictions = []
    for fold in folds:
        weights = fit_ridge(design_all[fold], targets[fold])
        predictions.append(design_all[holdout] @ weights)
    stack = np.stack(predictions)  # (folds, n_holdout, 2)
    mean = stack.mean(axis=0)
    variance = stack.std(axis=0).mean(axis=1)

    truth = targets[holdout]
    error = np.abs(mean - truth).mean(axis=1)
    predicted_cell = np.rint(mean * (size - 1)).astype(np.int64)
    true_cell = np.rint(truth * (size - 1)).astype(np.int64)
    hit = float(np.mean(np.all(predicted_cell == true_cell, axis=1)))

    counts = np.zeros(size * size * len(ACTIONS), dtype=np.int64)
    keys, key_counts = np.unique(states[train] * len(ACTIONS) + actions[train], return_counts=True)
    counts[keys] = key_counts
    holdout_counts = counts[states[holdout] * len(ACTIONS) + actions[holdout]]
    low = holdout_counts <= np.median(holdout_counts)
    coverage_ok = float(variance[low].mean()) > float(variance[~low].mean())
    spread = float(variance.std() / variance.mean()) if variance.mean() > 0 else 0.0

    return {
        "seed": seed,
        "holdout": int(holdout.sum()),
        "hit": hit,
        "spearman": spearman(variance.tolist(), error.tolist()),
        "coverage_ok": coverage_ok,
        "low_count_variance": float(variance[low].mean()),
        "high_count_variance": float(variance[~low].mean()),
        "variance_cv": spread,
    }


def e0_checks() -> dict:
    first, second = make_grid(7), make_grid(7)
    deterministic = (
        first.walls == second.walls and first.start == second.start and first.goal == second.goal
    )

    mismatches, checked = 0, 0
    for size in (4, 5):
        for seed in range(20):
            grid = make_grid(seed, size=size)
            checked += 1
            if bfs_length(grid) != brute_force_length(grid):
                mismatches += 1

    grids = [make_grid(seed) for seed in range(N_GRIDS)]
    lengths = [bfs_length(grid) for grid in grids]
    solvable = [length for length in lengths if length is not None]
    rate = len(solvable) / N_GRIDS

    playable = [grid for grid, length in zip(grids, lengths, strict=True) if length is not None]
    rates = {}
    for name in ("random", "greedy", "follow"):
        wins, total = 0, 0
        for index, grid in enumerate(playable):
            policy = (
                greedy_policy if name == "greedy" else random_policy if name == "random" else None
            )
            if name == "follow":
                policy = make_follow_policy(grid)
                if policy is None:
                    continue
            for seed in SEEDS:
                total += 1
                rng = random.Random(seed * 100_003 + index)
                wins += run_policy(grid, policy, rng, p_slip=DEFAULT_SLIP)
        rates[name] = wins / total if total else float("nan")

    return {
        "e0_1_deterministic": deterministic,
        "e0_2_bfs_checked": checked,
        "e0_2_bfs_mismatches": mismatches,
        "e0_3_solvable_rate": rate,
        "e0_4_distinct_lengths": len(set(solvable)),
        "e0_4_min_length": min(solvable) if solvable else None,
        "e0_4_max_length": max(solvable) if solvable else None,
        "e0_5_rates": rates,
        "pass": (
            deterministic
            and mismatches == 0
            and rate >= E0_SOLVABLE_MIN
            and len(set(solvable)) >= E0_DISTINCT_MIN
            and min(solvable) <= E0_SHORT_MAX
            and max(solvable) >= E0_LONG_MIN
            and rates["follow"] >= E0_FOLLOW_MIN
            and rates["follow"] - rates["random"] >= E0_GAP_MIN
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="脉冲 SoS 的 E0–E1")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="e0_e1")
    args = parser.parse_args(argv)

    e0 = e0_checks()
    print(
        f"[E0] 确定性 {e0['e0_1_deterministic']} | BFS 对照 {e0['e0_2_bfs_checked']} 张 "
        f"(不一致 {e0['e0_2_bfs_mismatches']}) | 可解率 {e0['e0_3_solvable_rate']:.3f} | "
        f"难度跨度 {e0['e0_4_distinct_lengths']} 档 [{e0['e0_4_min_length']}, {e0['e0_4_max_length']}]"
    )
    rates = e0["e0_5_rates"]
    print(
        f"[E0] 基线成功率：随机 {rates['random']:.3f} / 贪心 {rates['greedy']:.3f} / "
        f"BFS 跟随 {rates['follow']:.3f} | E0 {'过' if e0['pass'] else '不过'}"
    )

    result = {"kind": "spiking sos e0-e1", "e0": e0, "e1": None, "p1_ready": False}
    if e0["pass"]:
        grids = [make_grid(seed) for seed in range(TRAIN_GRIDS + HOLDOUT_GRIDS)]
        grids = [grid for grid in grids if bfs_length(grid) is not None]
        e1_rows = [evaluate_seed(seed, grids, grids[0].size) for seed in SEEDS]
        hits = statistics.median(row["hit"] for row in e1_rows)
        rhos = statistics.median(row["spearman"] for row in e1_rows)
        cvs = statistics.median(row["variance_cv"] for row in e1_rows)
        coverage = sum(1 for row in e1_rows if row["coverage_ok"])
        passed = (
            hits >= E1_HIT_MIN
            and rhos >= E1_SPEARMAN_MIN
            and coverage >= E1_COVERAGE_SEEDS_MIN
            and cvs >= E1_CV_MIN
        )
        result["e1"] = {
            "rows": e1_rows,
            "hit_median": hits,
            "spearman_median": rhos,
            "variance_cv_median": cvs,
            "coverage_seeds": coverage,
            "pass": passed,
        }
        result["p1_ready"] = passed
        print(
            f"[E1] 命中率中位 {hits:.3f} | ρ 中位 {rhos:.3f} | 方差 CV 中位 {cvs:.3f} | "
            f"覆盖度方向 {coverage}/5 种子 | P1 {'过' if passed else '不过'}"
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
