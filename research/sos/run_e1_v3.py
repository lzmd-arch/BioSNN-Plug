"""E1 v3：每迷宫在线世界模型（判据见 prereg-e0-e1-v3.md + amend1）。

模型 = `p(下一格 | 状态, 动作)`，**只从本迷宫的转移**里学；平滑的支撑集是几何上一步可达的
格子（自身 + 在界内的四个邻居），**不含墙**——墙正是模型要学的东西。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from random import Random

from .grid import ACTIONS, DEFAULT_SLIP, Grid, bfs_length, make_maze, step
from .run_e0_e1 import SEEDS, spearman

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "sweep_results" / "sos_e1_v3"

E1_SIZE = 11
N_GRIDS = 200
LEARN_FRACTION = 0.60
ALPHA = 1.0

#: 判据（改这里必须改预注册）。
E1_GAP_MIN = 0.20
E1_SPEARMAN_MIN = 0.30
E1_COVERAGE_SEEDS_MIN = 4
E1_CV_MIN = 0.10


def episode_cap(grid: Grid) -> int:
    length = bfs_length(grid)
    return 3 * length + 20 if length is not None else 60


def support(grid: Grid, cell: tuple[int, int]) -> list[tuple[int, int]]:
    cells = [cell]
    for dx, dy in ACTIONS:
        candidate = (cell[0] + dx, cell[1] + dy)
        if grid.inside(candidate):
            cells.append(candidate)
    return cells


def rollout(grid: Grid, seed: int) -> list[tuple[tuple[int, int], int, tuple[int, int]]]:
    rng = Random(seed)
    transitions = []
    cell = grid.start
    for _ in range(episode_cap(grid)):
        action = rng.randrange(len(ACTIONS))
        nxt = step(grid, cell, action, rng, p_slip=DEFAULT_SLIP)
        transitions.append((cell, action, nxt))
        cell = nxt
    return transitions


def build_table(rows) -> dict[tuple[tuple[int, int], int], Counter]:
    table: dict[tuple[tuple[int, int], int], Counter] = {}
    for cell, action, nxt in rows:
        table.setdefault((cell, action), Counter())[nxt] += 1
    return table


def smoothed(grid: Grid, counter: Counter | None, cell, action, target) -> float:
    cells = support(grid, cell)
    total = sum(counter.values()) if counter else 0
    hits = counter[target] if counter else 0
    return (hits + ALPHA) / (total + ALPHA * len(cells))


def predictive_entropy(grid: Grid, counter: Counter | None, cell, action) -> float:
    cells = support(grid, cell)
    probs = [smoothed(grid, counter, cell, action, target) for target in cells]
    return -sum(p * math.log(p) for p in probs if p > 0)


def evaluate(seed: int, grids: list[Grid]) -> dict:
    model_hits = marginal_hits = total = 0
    entropies: list[float] = []
    surprisals: list[float] = []
    coverage_flags: list[bool] = []
    for index, grid in enumerate(grids):
        rows = rollout(grid, seed * 10_007 + index)
        cut = int(len(rows) * LEARN_FRACTION)
        learn, evaluate_rows = rows[:cut], rows[cut:]
        if not evaluate_rows:
            continue
        table = build_table(learn)
        offsets: dict[int, Counter] = {}
        for cell, action, nxt in learn:
            offsets.setdefault(action, Counter())[(nxt[0] - cell[0], nxt[1] - cell[1])] += 1

        low_entropy: list[float] = []
        high_entropy: list[float] = []
        counts = [len(table.get((cell, action), ())) for cell, action, _ in evaluate_rows]
        median = statistics.median(counts)
        for (cell, action, nxt), count in zip(evaluate_rows, counts, strict=True):
            counter = table.get((cell, action))
            cells = support(grid, cell)
            best = max(cells, key=lambda target: smoothed(grid, counter, cell, action, target))
            model_hits += best == nxt
            # 边际基线：只看动作，取本迷宫最常见的位移，落在界内就用，否则原地。
            shift = (
                offsets.get(
                    action,
                    Counter(
                        (0, 0),
                    ),
                ).most_common(1)[0][0]
                if offsets.get(action)
                else (0, 0)
            )
            candidate = (cell[0] + shift[0], cell[1] + shift[1])
            marginal = candidate if grid.inside(candidate) else cell
            marginal_hits += marginal == nxt
            total += 1
            entropy = predictive_entropy(grid, counter, cell, action)
            entropies.append(entropy)
            surprisals.append(-math.log(max(smoothed(grid, counter, cell, action, nxt), 1e-12)))
            if count <= median:
                low_entropy.append(entropy)
            else:
                high_entropy.append(entropy)
        if low_entropy and high_entropy:
            coverage_flags.append(statistics.fmean(low_entropy) > statistics.fmean(high_entropy))

    return {
        "seed": seed,
        "n": total,
        "hit_model": model_hits / total,
        "hit_marginal": marginal_hits / total,
        "gap": (model_hits - marginal_hits) / total,
        "spearman": spearman(entropies, surprisals),
        "coverage_ok": bool(coverage_flags and sum(coverage_flags) * 2 > len(coverage_flags)),
        "coverage_grids": f"{sum(coverage_flags)}/{len(coverage_flags)}",
        "variance_cv": statistics.pstdev(entropies) / statistics.fmean(entropies)
        if statistics.fmean(entropies) > 0
        else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="脉冲 SoS 的 E1 v3（每迷宫在线世界模型）")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="e1_v3")
    args = parser.parse_args(argv)

    grids = [make_maze(seed, size=E1_SIZE) for seed in range(N_GRIDS)]
    rows = [evaluate(seed, grids) for seed in SEEDS]
    gap = statistics.median(row["gap"] for row in rows)
    rho = statistics.median(row["spearman"] for row in rows)
    cvs = statistics.median(row["variance_cv"] for row in rows)
    coverage = sum(1 for row in rows if row["coverage_ok"])
    passed = (
        gap >= E1_GAP_MIN
        and rho >= E1_SPEARMAN_MIN
        and coverage >= E1_COVERAGE_SEEDS_MIN
        and cvs >= E1_CV_MIN
    )
    result = {
        "kind": "spiking sos e1 v3 (per-maze online)",
        "size": E1_SIZE,
        "grids": N_GRIDS,
        "learn_fraction": LEARN_FRACTION,
        "alpha": ALPHA,
        "rows": rows,
        "gap_median": gap,
        "spearman_median": rho,
        "variance_cv_median": cvs,
        "coverage_seeds": coverage,
        "p1_ready": passed,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    for row in rows:
        print(
            f"[seed {row['seed']}] 命中 模型 {row['hit_model']:.3f} / 边际 {row['hit_marginal']:.3f} "
            f"= {row['gap']:+.3f} | ρ {row['spearman']:.3f} | 覆盖 {row['coverage_grids']} | "
            f"熵 CV {row['variance_cv']:.3f}"
        )
    print(
        f"[E1] gap 中位 {gap:+.3f}（≥{E1_GAP_MIN}）| ρ 中位 {rho:.3f}（≥{E1_SPEARMAN_MIN}）| "
        f"覆盖 {coverage}/5（≥{E1_COVERAGE_SEEDS_MIN}）| CV 中位 {cvs:.3f}（≥{E1_CV_MIN}）"
        f" | P1 {'过' if passed else '不过'}"
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
