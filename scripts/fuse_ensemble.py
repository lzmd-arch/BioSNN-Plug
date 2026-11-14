"""快半的**多截断集成**：同一位置用几个不同的上下文长度各跑一次，概率取平均。

动机：千问一次前向只给一个分布；快半便宜到能跑很多次。不同截断看到的信息不同，
误差会部分抵消——**用算力换去相关**。这是唯一不改数据、不改模型、只改推理的路径。

判据：集成的 F-0（千问错而快半对）要显著大于单次；同时不能把 top-1 拉低。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

FAST = "http://127.0.0.1:18734/complete"
TRUNCATIONS = (64, 128, 192, 256)


def post(url: str, payload: dict, timeout: float = 120.0) -> dict:
    import urllib.request

    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as handle:
        return json.loads(handle.read())


def fast_at(context: str, trunc: int, k: int = 10) -> list[tuple[str, float]]:
    body = post(FAST, {"context": context[-trunc:], "k": k})
    return [(c["token"], float(c["logprob"])) for c in body.get("candidates") or []]


def mixture(views: list[list[tuple[str, float]]]) -> dict[str, float]:
    """把多份 top-k 分布取平均（**概率空间**平均，再回到 log）。"""
    acc: dict[str, float] = defaultdict(float)
    for view in views:
        total = sum(math.exp(lp) for _, lp in view) or 1e-12
        for token, lp in view:
            acc[token] += math.exp(lp) / total
    n = max(len(views), 1)
    return {t: math.log(max(p / n, 1e-12)) for t, p in acc.items()}


def ok(tok: str, truth: str) -> bool:
    return bool(tok) and truth.startswith(tok)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="快半的多截断集成")
    ap.add_argument(
        "--cache", type=Path, default=Path("sweep_results/lm_fusion/alpha_v1_big.raw.json")
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    raw = json.loads(args.cache.read_text(encoding="utf-8"))
    if args.limit:
        raw = raw[: args.limit]

    single_fast_ok = ensemble_fast_ok = slow_ok = single_fast_only = ensemble_fast_only = 0
    n = 0
    for row in raw:
        slow, truth = row["slow"], row["truth"]
        if not slow:
            continue
        try:
            views = [fast_at(row["context"], t) for t in TRUNCATIONS]
        except Exception as exc:
            print(f"  跳过：{type(exc).__name__}", flush=True)
            continue
        n += 1
        s_best = max(slow, key=lambda kv: kv[1])[0]
        s_ok = ok(s_best, truth)
        slow_ok += int(s_ok)

        single = dict(max(views, key=len))
        f_single = max(single, key=lambda t: single[t])
        f_ens_table = mixture(views)
        f_ens = max(f_ens_table, key=lambda t: f_ens_table[t])

        single_fast_ok += int(ok(f_single, truth))
        ensemble_fast_ok += int(ok(f_ens, truth))
        single_fast_only += int(ok(f_single, truth) and not s_ok)
        ensemble_fast_only += int(ok(f_ens, truth) and not s_ok)

    print(f"\n位置数 {n}")
    print(f"  慢半 top-1 命中          {slow_ok:3d}  ({slow_ok / max(n, 1):.3f})")
    print(f"  快半单次 top-1 命中      {single_fast_ok:3d}  ({single_fast_ok / max(n, 1):.3f})")
    print(f"  快半**集成** top-1 命中  {ensemble_fast_ok:3d}  ({ensemble_fast_ok / max(n, 1):.3f})")
    print(f"\n  F-0 单次（千问错、快半对）  {single_fast_only}")
    print(f"  **F-0 集成**                {ensemble_fast_only}")
    oracle_single = slow_ok + single_fast_only
    oracle_ens = slow_ok + ensemble_fast_only
    print(
        f"\n  oracle 单次 {oracle_single} / 集成 {oracle_ens}"
        f"   ⇒ 集成多 {oracle_ens - oracle_single} 个位置"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
