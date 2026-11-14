"""唤醒通道的验收实验：规则在未知时刻切换的迷宫（预注册见 `prereg-arousal.md`）。

**任务**：每 K 个回合把「动作 → 方向」的映射做一次随机置换，不告知被试。
环境不变（同样的迷宫、同样的滑移），变的只有"按哪个键往哪走"——
所以策略必须重学，而重学需要探索。**这正是唤醒通道唯一可能证明自己的场景。**

**被试**：表格 Q-learning + softmax 动作选择。

    臂 A（固定）：τ ≡ 1.0
    臂 B（唤醒）：τ = exp(clip(β·(Ê − Ê_ref), 0, 2))

`Ê` 是 TD 误差的 EMA（一个**局部**量，不需要反向传播），`Ê_ref` 是上一段的均值。

用法::

    uv run python -m research.cognitive.run_arousal_switch --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

from research.sos.grid import make_maze, step

OUT = Path("sweep_results/arousal_switch")
SEGMENT = 40  # 每段回合数
SEGMENTS = 6  # 段数
N_ACTIONS = 4
P_SLIP = 0.10


def distance_map(grid) -> np.ndarray:
    """每个格子到终点的最短路长度（BFS）。塑形奖励用。"""
    size = grid.size
    dist = np.full((size, size), np.inf)
    dist[grid.goal[1], grid.goal[0]] = 0.0
    queue = [grid.goal]
    while queue:
        x, y = queue.pop(0)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if grid.blocked((nx, ny)) or dist[ny, nx] != np.inf:
                continue
            dist[ny, nx] = dist[y, x] + 1
            queue.append((nx, ny))
    finite = dist[np.isfinite(dist)]
    dist[np.isinf(dist)] = finite.max() if finite.size else 0.0
    return dist


def episode(
    grid,
    q: np.ndarray,
    perm: list[int],
    rng: random.Random,
    tau: float,
    *,
    learn: bool = True,
    gamma: float = 0.95,
    dist: np.ndarray | None = None,
    shaping: float = 1.0,
    alpha: float = 0.2,
) -> tuple[bool, float]:
    """跑一回合。返回 (是否到达终点, 平均 |TD 误差|)。"""
    size = grid.size
    cell = grid.start
    cap = 3 * (2 * size) + 20
    td_sum = 0.0
    steps = 0
    for _ in range(cap):
        state = cell[1] * size + cell[0]
        logits = q[state] / max(tau, 1e-6)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        action = int(rng.choices(range(N_ACTIONS), weights=probs.tolist())[0])
        nxt = step(grid, cell, perm[action], rng, p_slip=P_SLIP)
        if nxt == grid.goal:
            reward = 1.0
        elif dist is not None:
            # **势函数塑形**（potential-based）：Φ(s) = −dist(s)/max_dist，
            # r' = γΦ(s') − Φ(s)。它保证最优策略不变，而且**不会让 TD 误差恒为 0**。
            # 第一版写成"−0.05·dist"，配合零初始化 Q 时 target = γ·0 = 0、error = 0，
            # 表格永远不动——实测 120 回合 |Q|max 一直是 0.000。
            potential_now = -float(dist[cell[1], cell[0]]) / float(dist.max())
            potential_next = -float(dist[nxt[1], nxt[0]]) / float(dist.max())
            reward = shaping * (gamma * potential_next - potential_now)
        else:
            reward = 0.0
        nxt_state = nxt[1] * size + nxt[0]
        target = reward if reward > 0 else gamma * float(q[nxt_state].max())
        error = target - float(q[state, action])
        td_sum += abs(error)
        steps += 1
        if learn:
            q[state, action] += alpha * error
        cell = nxt
        if reward > 0:
            return True, td_sum / steps
    return False, td_sum / max(steps, 1)


def run_arm(
    seed: int,
    arm: str,
    beta: float = 4.0,
    base_tau: float = 1.0,
    shaping: float = 1.0,
    shaping_first_only: bool = False,
    alpha: float = 0.2,
) -> dict:
    grid = make_maze(seed, size=9)
    dist = distance_map(grid)
    q = np.zeros((grid.size * grid.size, N_ACTIONS))
    rng = random.Random(seed)
    perm = list(range(N_ACTIONS))
    successes: list[int] = []
    recovery: list[int] = []
    taus: list[float] = []
    error_ema = 0.0
    ref = 0.0
    for segment in range(SEGMENTS):
        # 恢复判据的**基线是切换前那一段**的达成率，不是第一段的。
        baseline_rate = (
            float(np.mean(successes[(segment - 1) * SEGMENT : segment * SEGMENT]))
            if segment > 0
            else 0.0
        )
        if segment > 0:
            rng.shuffle(perm)  # 规则切换：动作映射被置换
            ref = error_ema
        segment_success = []
        recovered_at = None
        for index in range(SEGMENT):
            if arm == "arousal":
                arousal = min(max(beta * (error_ema - ref), 0.0), 2.0)
                tau = base_tau * math.exp(arousal)
            else:
                tau = base_tau
            taus.append(tau)
            scale = shaping if (segment == 0 or not shaping_first_only) else 0.0
            ok, td = episode(grid, q, perm, rng, tau, dist=dist, shaping=scale, alpha=alpha)
            successes.append(int(ok))
            segment_success.append(int(ok))
            error_ema = 0.95 * error_ema + 0.05 * td
            if segment > 0 and recovered_at is None and len(segment_success) >= 5:
                trailing = np.mean(segment_success[-5:])
                if trailing >= 0.8 * baseline_rate:
                    recovered_at = index + 1
        if segment > 0:
            recovery.append(recovered_at if recovered_at is not None else SEGMENT)
    return {
        "seed": seed,
        "arm": arm,
        "success_rate": float(np.mean(successes)),
        "segment_end_success": float(
            np.mean([np.mean(successes[i * SEGMENT : (i + 1) * SEGMENT]) for i in range(SEGMENTS)])
        ),
        "recovery_median": float(np.median(recovery)) if recovery else float(SEGMENT),
        "tau_max": float(np.max(taus)),
        "first_segment_success": float(np.mean(successes[:SEGMENT])),
        "second_segment_first5": float(np.mean(successes[SEGMENT : SEGMENT + 5])),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="唤醒通道：规则切换的迷宫")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--base-tau", type=float, default=0.25)
    ap.add_argument("--shaping", type=float, default=1.0)
    ap.add_argument("--shaping-first-only", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows = []
    for seed in seeds:
        kw = {
            "base_tau": args.base_tau,
            "shaping": args.shaping,
            "shaping_first_only": args.shaping_first_only,
        }
        fixed = run_arm(seed, "fixed", **kw)
        arousal = run_arm(seed, "arousal", beta=args.beta, **kw)
        rows.extend([fixed, arousal])
        print(
            f"种子 {seed}: 固定 段尾 {fixed['segment_end_success']:.2f} 恢复 {fixed['recovery_median']:.0f}"
            f" | 唤醒 段尾 {arousal['segment_end_success']:.2f} 恢复 {arousal['recovery_median']:.0f}"
            f"（τ_max {arousal['tau_max']:.2f}）",
            flush=True,
        )

    def med(arm: str, key: str) -> float:
        return float(np.median([r[key] for r in rows if r["arm"] == arm]))

    fixed_rec = med("fixed", "recovery_median")
    arousal_rec = med("arousal", "recovery_median")
    fixed_end = med("fixed", "segment_end_success")
    arousal_end = med("arousal", "segment_end_success")
    drop = med("fixed", "first_segment_success") - med("fixed", "second_segment_first5")

    print(
        f"\n中位恢复回合数：固定 {fixed_rec:.0f} / 唤醒 {arousal_rec:.0f}"
        f"（比值 {arousal_rec / max(fixed_rec, 1e-9):.2f}）"
    )
    print(f"中位段尾成功率：固定 {fixed_end:.2f} / 唤醒 {arousal_end:.2f}")
    print(
        f"AR-0b 切换后 5 回合成功率跌幅：{drop:.2f} ⇒ {'过' if drop >= 0.10 else '不过（没感知到切换）'}"
    )
    print(f"AR-1a 恢复更快（≤0.8×）：{'过' if arousal_rec <= 0.8 * fixed_rec else '不过'}")
    print(
        f"AR-1b 平稳段不掉（唤醒 ≥ 固定 − 0.01）：{'过' if arousal_end >= fixed_end - 0.01 else '不过'}"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "segment": SEGMENT,
                "segments": SEGMENTS,
                "beta": args.beta,
                "rows": rows,
                "fixed_recovery": fixed_rec,
                "arousal_recovery": arousal_rec,
                "fixed_end": fixed_end,
                "arousal_end": arousal_end,
                "switch_drop": drop,
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"写入 {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
