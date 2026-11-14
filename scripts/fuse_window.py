"""V1 的**输入侧**扫描：快半到底该看多长的上下文？（不重训，只改运行时参数）

动机：8 字上下文上快半 top-1 命中 3/52，20–256 字上 **0/543**。
这说明它的有效记忆很短，长上下文可能在稀释它。窗口是运行时参数，不碰权重。

同时给出产品真用得上的那个指标：**候选补全的 top-5 召回**——
用户从几个候选里挑，快半只要能补上慢半漏掉的，就是能力增益，不需要任何仲裁器。

协议（判据先写死）：

1. 位置与切分同 `prereg-fusion-arbiter-v1.md`（按记忆条目时间切，聚类 bootstrap）；
2. 窗口在**训练集**上选（top-1 命中率最高者），**留出集只报数**；
3. 报每个窗口：快半 top-1 / top-5、并集 top-5 召回、oracle、α 扫描；
4. 判据 **W-1**：留出集上并集 top-5 召回 ≥ 慢半单独；**W-2**：留出集 oracle 是否 > 慢半。

若所有窗口的 oracle 都等于慢半 ⇒ 结论是"快半在长上下文上不提供任何可用的候选"，
V1 的候选补全只能靠 8 字级短上下文。
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

SLOW = "http://127.0.0.1:8081/v1/completions"
FAST = "http://127.0.0.1:18762/complete"
DB = Path("D:/AxoData/memory/mem.db")
WINDOWS = (4, 8, 16, 32, 64, 128, 256)
ALPHAS = (0.0, 0.03, 0.05, 0.1, 0.25)


def post(url: str, payload: dict, timeout: float = 120.0) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as handle:
        return json.loads(handle.read())


def read_memories(db: Path, min_len: int) -> list[str]:
    uri = "file:" + urllib.parse.quote(str(db).replace("\\", "/")) + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = con.execute(
            "select content from memories where content is not null and length(content) > ? "
            "order by created_at",
            (min_len,),
        ).fetchall()
    finally:
        con.close()
    return [row[0] for row in rows]


def hits(token: str, truth: str) -> bool:
    return bool(token) and truth.startswith(token)


def fuse_table(slow, fast, alpha: float) -> dict[str, float]:
    if not slow:
        return {}
    if not fast or alpha == 0.0:
        return dict(slow)
    floor = min(lp for _, lp in slow) - 5.0
    table: dict[str, float] = {}
    for token, logprob in slow:
        mass = sum(math.exp(slp) for stoken, slp in fast if stoken and token.startswith(stoken))
        table[token] = logprob + alpha * math.log(max(mass, 1e-12))
    for stoken, slp in fast:
        if stoken and not any(token.startswith(stoken) for token, _ in slow):
            table.setdefault(stoken, floor + alpha * slp)
    return table


def _best(table: dict[str, float]) -> str:
    return max(table.items(), key=lambda kv: kv[1])[0] if table else ""


def mass_log(table: dict[str, float], truth: str) -> float:
    total = sum(math.exp(v) for v in table.values()) or 1e-12
    hit = sum(math.exp(v) for t, v in table.items() if t and truth.startswith(t))
    return math.log(max(hit / total, 1e-12))


def cluster_ci(values: np.ndarray, groups: list[str], reps: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    index: dict[str, list[int]] = {}
    for row, key in enumerate(groups):
        index.setdefault(key, []).append(row)
    clusters = list(index.values())
    means = np.empty(reps)
    for rep in range(reps):
        picks = rng.integers(0, len(clusters), size=len(clusters))
        rows = np.concatenate([clusters[c] for c in picks])
        means[rep] = values[rows].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def topk_recall(cands: list[tuple[str, float]], truth: str, k: int) -> bool:
    return any(hits(token, truth) for token, _ in cands[:k])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="快半窗口扫描（V1，不重训）")
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--min-len", type=int, default=24)
    ap.add_argument("--per-entry", type=int, default=16)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--out", type=Path, default=Path("sweep_results/lm_fusion/window_v1.json"))
    args = ap.parse_args(argv)

    texts = read_memories(args.db, args.min_len)
    cut = int(len(texts) * args.train_frac)
    splits = {"train": texts[:cut], "holdout": texts[cut:]}
    print(f"记忆 {len(texts)} 条：训练 {cut} / 留出 {len(texts) - cut}", flush=True)

    rows: list[dict] = []
    skipped = 0
    for split, entries in splits.items():
        for entry_id, text in enumerate(entries):
            for start in list(range(20, max(len(text) - 2, 21), args.stride))[: args.per_entry]:
                context, truth = text[:start], text[start:]
                try:
                    slow = [
                        (i["token"], float(i["logprob"]))
                        for i in post(
                            SLOW,
                            {
                                "model": "qwen3",
                                "prompt": context,
                                "max_tokens": 1,
                                "temperature": 0,
                                "logprobs": 20,
                            },
                        )["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                    ]
                    fast = {
                        window: [
                            (c["token"], float(c["logprob"]))
                            for c in post(FAST, {"context": context[-window:], "k": 10})[
                                "candidates"
                            ]
                        ]
                        for window in WINDOWS
                    }
                except Exception as exc:
                    skipped += 1
                    print(f"  跳过：{type(exc).__name__}", flush=True)
                    continue
                if not slow or not all(fast.values()):
                    skipped += 1
                    continue
                rows.append(
                    {
                        "split": split,
                        "entry": f"{split}-{entry_id}",
                        "context_len": len(context),
                        "truth": truth[:8],
                        "slow": slow,
                        "fast": fast,
                    }
                )
        print(f"  {split}: 累计 {len(rows)} 个位置", flush=True)

    train = [r for r in rows if r["split"] == "train"]
    hold = [r for r in rows if r["split"] == "holdout"]
    groups = [r["entry"] for r in hold]
    print(f"\n位置：训练 {len(train)} / 留出 {len(hold)}，跳过 {skipped}")

    def evaluate(window: int, sample: list[dict]) -> dict:
        hit1 = np.array([float(topk_recall(r["fast"][window], r["truth"], 1)) for r in sample])
        hit5 = np.array([float(topk_recall(r["fast"][window], r["truth"], 5)) for r in sample])
        slow1 = np.array([float(hits(r["slow"][0][0], r["truth"])) for r in sample])
        slow5 = np.array([float(topk_recall(r["slow"], r["truth"], 5)) for r in sample])
        union5 = np.array(
            [
                float(
                    topk_recall(r["slow"], r["truth"], 5)
                    or topk_recall(r["fast"][window], r["truth"], 5)
                )
                for r in sample
            ]
        )
        return {
            "fast_top1": round(float(hit1.mean()), 4),
            "fast_top5": round(float(hit5.mean()), 4),
            "slow_top1": round(float(slow1.mean()), 4),
            "slow_top5": round(float(slow5.mean()), 4),
            "union_top5": round(float(union5.mean()), 4),
            "oracle_top1": round(float(np.maximum(slow1, hit1).mean()), 4),
            "n": len(sample),
        }

    train_stats = {w: evaluate(w, train) for w in WINDOWS}
    print("\n训练集（窗口在这里选）：")
    for w, s in train_stats.items():
        print(
            f"  w={w:>3}  快半 top-1 {s['fast_top1']:.4f}  top-5 {s['fast_top5']:.4f}"
            f"  并集 top-5 {s['union_top5']:.4f}  oracle {s['oracle_top1']:.4f}"
        )

    # **窗口只在训练集上选**：top-1 与并集召回各选一个
    best_by_top1 = max(WINDOWS, key=lambda w: train_stats[w]["fast_top1"])
    best_by_union = max(WINDOWS, key=lambda w: train_stats[w]["union_top5"])
    print(f"\n训练集选出：top-1 最优窗口 {best_by_top1}；并集召回最优窗口 {best_by_union}")

    hold_stats = {w: evaluate(w, hold) for w in WINDOWS}
    print("\n留出集（只报数）：")
    deltas = {}
    for w, s in hold_stats.items():
        d5 = np.array(
            [
                float(
                    topk_recall(r["slow"], r["truth"], 5)
                    or topk_recall(r["fast"][w], r["truth"], 5)
                )
                - float(topk_recall(r["slow"], r["truth"], 5))
                for r in hold
            ]
        )
        lo, hi = cluster_ci(d5, groups)
        deltas[w] = {
            "union_minus_slow_top5_pp": round(float(d5.mean()) * 100, 2),
            "ci95": [round(lo * 100, 2), round(hi * 100, 2)],
        }
        print(
            f"  w={w:>3}  快半 top-1 {s['fast_top1']:.4f}  top-5 {s['fast_top5']:.4f}"
            f"  并集 top-5 {s['union_top5']:.4f} ({deltas[w]['union_minus_slow_top5_pp']:+.2f}pp,"
            f" CI {deltas[w]['ci95']})  oracle {s['oracle_top1']:.4f}"
        )

    # α 扫描：在留出集上，对两个候选窗口
    sweep = {}
    base_logs = np.array([mass_log(dict(r["slow"]), r["truth"]) for r in hold])
    base_top1 = np.array([float(hits(r["slow"][0][0], r["truth"])) for r in hold])
    for window in sorted({best_by_top1, best_by_union}):
        rows_a = []
        for alpha in ALPHAS:
            logs = np.array(
                [
                    mass_log(fuse_table(r["slow"], r["fast"][window], alpha), r["truth"])
                    for r in hold
                ]
            )
            tops = np.array(
                [
                    float(hits(_best(fuse_table(r["slow"], r["fast"][window], alpha)), r["truth"]))
                    for r in hold
                ]
            )
            lo_l, hi_l = cluster_ci(logs - base_logs, groups)
            lo_t, hi_t = cluster_ci(tops - base_top1, groups)
            rows_a.append(
                {
                    "alpha": alpha,
                    "delta_logp": round(float((logs - base_logs).mean()), 4),
                    "delta_logp_ci95": [round(lo_l, 4), round(hi_l, 4)],
                    "delta_top1_pp": round(float((tops - base_top1).mean()) * 100, 2),
                    "delta_top1_ci95_pp": [round(lo_t * 100, 2), round(hi_t * 100, 2)],
                }
            )
        sweep[str(window)] = rows_a
        print(f"\n窗口 {window} 上的 α 扫描（留出集）：")
        for row in rows_a:
            print(
                f"  α={row['alpha']:<5} Δlog p {row['delta_logp']:+.4f}"
                f" CI {row['delta_logp_ci95']}  Δtop-1 {row['delta_top1_pp']:+.2f}pp"
                f" CI {row['delta_top1_ci95_pp']}"
            )

    result = {
        "windows": list(WINDOWS),
        "positions": {"train": len(train), "holdout": len(hold), "skipped": skipped},
        "train": {str(w): train_stats[w] for w in WINDOWS},
        "holdout": {str(w): hold_stats[w] for w in WINDOWS},
        "holdout_union_delta": {str(w): deltas[w] for w in WINDOWS},
        "chosen": {"by_top1": best_by_top1, "by_union": best_by_union},
        "alpha_sweep": sweep,
        "context_len": {
            "min": min(r["context_len"] for r in hold),
            "max": max(r["context_len"] for r in hold),
            "mean": round(float(np.mean([r["context_len"] for r in hold])), 1),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\n结果 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
