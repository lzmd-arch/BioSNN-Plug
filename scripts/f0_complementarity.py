"""F-0（互补性）：千问单独 vs 快半单独，谁在哪栽跟头？

要"比千问 4B 强"，唯一的前提是**快半能在千问错的地方对**。用缓存算，不跑前向。

三种口径同时给，因为它们回答不同的问题：

    slow_top1    慢半 top-1 是否命中（这就是"千问单独"的能力）
    fast_topk    慢半 top-k 里是否有一个是正确的**前缀**（快半还救得回来吗）
    oracle       两者只要有一个对就算对（**理论天花板**：完美仲裁能拿到多少）

oracle 与 slow_top1 之差，就是"融合最多还能赚多少"。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def correct(tokens: list[str], truth: str) -> bool:
    return any(t and truth.startswith(t) for t in tokens)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F-0：两半的互补性")
    ap.add_argument(
        "--cache",
        type=Path,
        default=Path("sweep_results/lm_fusion/alpha_v1_big.raw.json"),
    )
    args = ap.parse_args(argv)

    raw = json.loads(args.cache.read_text(encoding="utf-8"))
    n = len(raw)
    stats = {"slow_ok": 0, "fast_ok": 0, "both_ok": 0, "slow_only": 0, "fast_only": 0, "none": 0}
    uncertain_rows = []
    for row in raw:
        slow, fast, truth = row["slow"], row["fast"], row["truth"]
        s = correct([t for t, _ in slow[:1]], truth)
        f = correct([t for t, _ in fast[:1]], truth)
        stats["slow_ok"] += int(s)
        stats["fast_ok"] += int(f)
        stats["both_ok"] += int(s and f)
        stats["slow_only"] += int(s and not f)
        stats["fast_only"] += int(f and not s)
        stats["none"] += int(not s and not f)
        # 慢半不确定的子集：top-1 的 logprob 低于阈值时，快半有没有帮上忙
        if slow:
            top_lp = max(lp for _, lp in slow)
            if top_lp < -2.0:
                uncertain_rows.append({"slow_ok": s, "fast_ok": f, "top_lp": top_lp})

    oracle = n - stats["none"]
    print(f"位置数 {n}\n")
    print(f"  慢半 top-1 命中      {stats['slow_ok']:3d} / {n}   ({stats['slow_ok'] / n:.3f})")
    print(f"  快半 top-1 命中      {stats['fast_ok']:3d} / {n}   ({stats['fast_ok'] / n:.3f})")
    print(f"    ├ 两边都对          {stats['both_ok']:3d}")
    print(f"    ├ 只有慢半对        {stats['slow_only']:3d}")
    print(f"    └ **只有快半对**    {stats['fast_only']:3d}   ← F-0")
    print(f"  两边都错            {stats['none']:3d}")
    print(f"\n  **oracle 上限**      {oracle:3d} / {n}   ({oracle / n:.3f})")
    print(
        f"  相对慢半单独的空间   {oracle - stats['slow_ok']:+d} 个位置"
        f"（{(oracle - stats['slow_ok']) / n * 100:+.1f}pp）"
    )

    if uncertain_rows:
        u = len(uncertain_rows)
        u_slow = sum(r["slow_ok"] for r in uncertain_rows)
        u_fast = sum(r["fast_ok"] for r in uncertain_rows)
        print(f"\n  慢半不确定子集（top-1 logprob < −2.0）：{u} 个位置")
        print(f"    慢半命中 {u_slow} / 快半命中 {u_fast}")
    else:
        print("\n  没有慢半不确定的位置（阈值下无样本）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
