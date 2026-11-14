"""在缓存的原始响应上扫融合**变体**，不再重跑 Qwen 前向。

基线的问题是明确的：α>0 让真值似然变好，但 top-1 变差。
四个候选改法，都只动"怎么融合"，不动两半本身：

    A 细网格        α ∈ {0, 0.02, 0.03, 0.05, 0.07, 0.1}
    B 快半置信门    快半的 token 只有 logprob > τ 才参与（弱候选不进）
    C 慢半不确定门  只有慢半的熵 > H 时才融合（它确定时别打扰）
    D top-1 冻结    保留慢半的首选，只让快半重排第 2 名之后
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def mass(table: dict[str, float], truth: str) -> float:
    total = sum(math.exp(v) for v in table.values()) or 1e-12
    hit = sum(math.exp(v) for t, v in table.items() if t and truth.startswith(t))
    return math.log(max(hit / total, 1e-12))


def fuse(slow, fast, alpha: float, *, fast_tau: float | None = None) -> dict[str, float]:
    if not slow:
        return {}
    if not fast or alpha == 0.0:
        return dict(slow)
    if fast_tau is not None:
        fast = [(t, lp) for t, lp in fast if lp >= fast_tau]
    floor = min(lp for _, lp in slow) - 5.0
    table: dict[str, float] = {}
    for token, logprob in slow:
        m = sum(math.exp(slp) for stoken, slp in fast if stoken and token.startswith(stoken))
        table[token] = logprob + alpha * math.log(max(m, 1e-12))
    for stoken, slp in fast:
        if stoken and not any(token.startswith(stoken) for token, _ in slow):
            table.setdefault(stoken, floor + alpha * slp)
    return table


def entropy(slow) -> float:
    total = sum(math.exp(lp) for _, lp in slow) or 1e-12
    return -sum(
        (math.exp(lp) / total) * math.log(max(math.exp(lp) / total, 1e-12)) for _, lp in slow
    )


def evaluate(raw, build) -> tuple[float, float]:
    total_logp = 0.0
    hits = 0
    for row in raw:
        table = build(row["slow"], row["fast"], row["truth"])
        if not table:
            continue
        total_logp += mass(table, row["truth"])
        best = max(table.items(), key=lambda kv: kv[1])[0]
        hits += int(bool(best) and row["truth"].startswith(best))
    n = max(len(raw), 1)
    return total_logp / n, hits / n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="融合变体扫描（用缓存）")
    ap.add_argument(
        "--cache", type=Path, default=Path("sweep_results/lm_fusion/alpha_v1_cache.raw.json")
    )
    args = ap.parse_args(argv)

    raw = json.loads(args.cache.read_text(encoding="utf-8"))
    print(f"{len(raw)} 个位置\n")
    print(f"{'变体':<34}{'mean log p':>12}{'top-1':>9}")
    print("-" * 55)

    def show(label: str, build) -> None:
        logp, top1 = evaluate(raw, build)
        print(f"{label:<34}{logp:>12.4f}{top1:>9.4f}")

    show("A α=0（基线）", lambda s, f, t: fuse(s, f, 0.0))
    for alpha in (0.02, 0.03, 0.05, 0.07, 0.1):
        show(f"A α={alpha}", lambda s, f, t, a=alpha: fuse(s, f, a))

    for tau in (-4.0, -3.0, -2.0):
        show(f"B 快半置信门 τ={tau} (α=0.05)", lambda s, f, t, k=tau: fuse(s, f, 0.05, fast_tau=k))

    for h in (1.5, 2.0, 2.5):
        show(
            f"C 熵门 H>{h} (α=0.05)",
            lambda s, f, t, hh=h: fuse(s, f, 0.05) if entropy(s) > hh else fuse(s, f, 0.0),
        )

    def freeze(s, f, t):
        table = fuse(s, f, 0.05)
        if not table:
            return table
        slow_best = max(dict(s).items(), key=lambda kv: kv[1])[0]
        order = sorted(table.items(), key=lambda kv: (kv[0] != slow_best, -kv[1]))
        return dict(order)

    show("D top-1 冻结 (α=0.05)", freeze)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
