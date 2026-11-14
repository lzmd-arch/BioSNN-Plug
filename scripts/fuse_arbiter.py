"""学出来的仲裁器能不能抓住 F-0——判据见 `research/sel/prereg-fusion-arbiter-v1.md`。

按预注册的口径：按**条目**时间切训练/留出，特征集先写死，逻辑回归 + 训练集选阈值，
留出集上只报数与区间。**留出集不参与任何选择。**

用法::

    python scripts/fuse_arbiter.py --out sweep_results/lm_fusion/arbiter_v1.json
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

#: 预注册写死的特征集，**不许看结果再挑**。
FEATURES = (
    "slow_top1_logp",
    "slow_margin",
    "fast_top1_logp",
    "fast_margin",
    "agree",
    "fast_in_slow",
    "slow_logp_of_fast_minus_top1",
)


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


def prefix_overlap(a: str, b: str) -> bool:
    return bool(a) and bool(b) and (a.startswith(b) or b.startswith(a))


def fuse_table(slow: list[tuple[str, float]], fast: list[tuple[str, float]], alpha: float):
    """与 `fusion/semprea_fuse.py` 同一条公式。"""
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


def best_of(table: dict[str, float]) -> str:
    return max(table.items(), key=lambda kv: kv[1])[0] if table else ""


def features(slow, fast) -> list[float]:
    if not slow or not fast:
        return []
    s1, s2 = slow[0][1], (slow[1][1] if len(slow) > 1 else slow[0][1] - 10.0)
    f1, f2 = fast[0][1], (fast[1][1] if len(fast) > 1 else fast[0][1] - 10.0)
    fast_top = fast[0][0]
    # 不在慢半表里 ⇒ 按预注册填 −20（"它根本没被考虑过"）
    in_table = next((lp for token, lp in slow if prefix_overlap(token, fast_top)), None)
    return [
        s1,
        s1 - s2,
        f1,
        f1 - f2,
        float(prefix_overlap(slow[0][0], fast_top)),
        float(in_table is not None),
        (-20.0 if in_table is None else in_table) - s1,
    ]


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 4000, lr: float = 0.2):
    """带类别平衡加权的 L2 逻辑回归。**固定种子 ⇒ 可复现。**"""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-9] = 1.0
    z = (x - mean) / std
    z = np.hstack([z, np.ones((z.shape[0], 1))])
    w = np.zeros(z.shape[1])
    pos = max(y.sum(), 1.0)
    neg = max(len(y) - y.sum(), 1.0)
    weight = np.where(y > 0, 0.5 * len(y) / pos, 0.5 * len(y) / neg)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(z @ w)))
        grad = (z.T @ ((p - y) * weight)) / len(y) + l2 * np.concatenate([w[:-1], [0.0]]) / len(y)
        w -= lr * grad
    return {"w": w.tolist(), "mean": mean.tolist(), "std": std.tolist()}


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    w = np.asarray(model["w"])
    mean = np.asarray(model["mean"])
    std = np.asarray(model["std"])
    z = (x - mean) / std
    z = np.hstack([z, np.ones((z.shape[0], 1))])
    return 1.0 / (1.0 + np.exp(-(z @ w)))


def bootstrap_delta(
    a: np.ndarray,
    b: np.ndarray,
    groups: list[str],
    reps: int = 4000,
    seed: int = 0,
) -> tuple[float, float]:
    """**按记忆聚类的配对 bootstrap**：重采样条目，条目内位置一起进出。

    按位置重采样是错的——同一条记忆里相邻位置几乎同一句话，命中与否高度相关，
    那样算出来的区间偏窄，会把噪声读成显著。聚类重采样只会把区间放大，是更严的判据。
    """
    rng = np.random.default_rng(seed)
    diff = a.astype(float) - b.astype(float)
    index: dict[str, list[int]] = {}
    for row, key in enumerate(groups):
        index.setdefault(key, []).append(row)
    clusters = list(index.values())
    means = np.empty(reps)
    for rep in range(reps):
        picks = rng.integers(0, len(clusters), size=len(clusters))
        rows = np.concatenate([clusters[choice] for choice in picks])
        means[rep] = diff[rows].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="融合仲裁器（预注册：prereg-fusion-arbiter-v1）")
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--min-len", type=int, default=24)
    ap.add_argument("--per-entry", type=int, default=16)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--out", type=Path, default=Path("sweep_results/lm_fusion/arbiter_v1.json"))
    args = ap.parse_args(argv)

    texts = read_memories(args.db, args.min_len)
    cut = int(len(texts) * args.train_frac)
    splits = {"train": texts[:cut], "holdout": texts[cut:]}
    print(f"记忆 {len(texts)} 条：训练 {cut} / 留出 {len(texts) - cut}", flush=True)

    rows: list[dict] = []
    skipped = 0
    for split, entries in splits.items():
        for entry_id, text in enumerate(entries):
            starts = list(range(20, max(len(text) - 2, 21), args.stride))[: args.per_entry]
            for start in starts:
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
                    fast = [
                        (c["token"], float(c["logprob"]))
                        for c in post(FAST, {"context": context[-args.window :], "k": 10})[
                            "candidates"
                        ]
                    ]
                except Exception as exc:
                    skipped += 1
                    print(f"  跳过：{type(exc).__name__}", flush=True)
                    continue
                if not slow or not fast:
                    skipped += 1
                    continue
                row = {
                    "split": split,
                    "entry": f"{split}-{entry_id}",
                    "context_len": len(context),
                    "slow_top1": slow[0][0],
                    "fast_top1": fast[0][0],
                    "hit_slow": int(hits(slow[0][0], truth)),
                    "hit_fast": int(hits(fast[0][0], truth)),
                    "hit_fusion": int(hits(best_of(fuse_table(slow, fast, 0.03)), truth)),
                    "features": features(slow, fast),
                }
                rows.append(row)
        print(f"  {split}: 累计 {len(rows)} 个位置", flush=True)

    total = len(rows) + skipped
    skip_rate = skipped / max(total, 1)
    train = [r for r in rows if r["split"] == "train"]
    hold = [r for r in rows if r["split"] == "holdout"]
    print(f"\n位置：训练 {len(train)} / 留出 {len(hold)}，跳过率 {skip_rate:.1%}")

    result: dict = {"positions": {"train": len(train), "holdout": len(hold)}, "skipped": skipped}
    if not train or not hold or skip_rate > 0.2 or len(hold) < 500:
        result["verdict"] = "Q-0 不过：样本不足或跳过率过高，本轮作废"
        print(result["verdict"])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        return 1

    x_train = np.asarray([r["features"] for r in train])
    y_train = np.asarray([r["hit_fast"] and not r["hit_slow"] for r in train], dtype=float)
    model = fit_logistic(x_train, y_train)

    p_train = predict(model, x_train)
    hit_slow_train = np.asarray([r["hit_slow"] for r in train], dtype=float)
    hit_fast_train = np.asarray([r["hit_fast"] for r in train], dtype=float)
    best_thr, best_acc = 0.5, -1.0
    for thr in np.unique(np.round(p_train, 4)):
        pick_fast = p_train > thr
        acc = float(np.where(pick_fast, hit_fast_train, hit_slow_train).mean())
        if acc > best_acc:
            best_thr, best_acc = float(thr), acc
    print(f"阈值（训练集上选）= {best_thr:.4f}，训练集 top-1 = {best_acc:.4f}")

    x_hold = np.asarray([r["features"] for r in hold])
    p_hold = predict(model, x_hold)
    hit_slow = np.asarray([r["hit_slow"] for r in hold], dtype=float)
    hit_fast = np.asarray([r["hit_fast"] for r in hold], dtype=float)
    hit_fusion = np.asarray([r["hit_fusion"] for r in hold], dtype=float)
    hit_arb = np.where(p_hold > best_thr, hit_fast, hit_slow)
    oracle = np.maximum(hit_slow, hit_fast)

    groups = [r["entry"] for r in hold]
    lo, hi = bootstrap_delta(hit_arb, hit_slow, groups)
    lo_f, hi_f = bootstrap_delta(hit_fusion, hit_slow, groups)
    summary = {
        "train_positions": len(train),
        "holdout_positions": len(hold),
        "threshold": round(best_thr, 4),
        "holdout": {
            "slow_alone": round(float(hit_slow.mean()), 4),
            "fast_alone": round(float(hit_fast.mean()), 4),
            "fusion_alpha003": round(float(hit_fusion.mean()), 4),
            "arbiter": round(float(hit_arb.mean()), 4),
            "oracle": round(float(oracle.mean()), 4),
            "delta_arbiter_vs_slow_pp": round(float((hit_arb - hit_slow).mean()) * 100, 2),
            "delta_arbiter_ci95_pp": [round(lo * 100, 2), round(hi * 100, 2)],
            "delta_fusion_vs_slow_pp": round(float((hit_fusion - hit_slow).mean()) * 100, 2),
            "delta_fusion_ci95_pp": [round(lo_f * 100, 2), round(hi_f * 100, 2)],
            "switched": int((p_hold > best_thr).sum()),
            "switched_and_right": int(
                ((p_hold > best_thr) & (hit_fast > 0) & (hit_slow == 0)).sum()
            ),
            "switched_and_wrong": int(
                ((p_hold > best_thr) & (hit_slow > 0) & (hit_fast == 0)).sum()
            ),
        },
        "features": list(FEATURES),
        "model": model,
    }
    result.update(summary)
    result["verdict"] = "Q-3 过：区间下界 > 0" if lo > 0 else "Q-3 不过：区间含 0，没有可用的增益"
    print(json.dumps(summary["holdout"], ensure_ascii=False, indent=1))
    print(f"\n{result['verdict']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    cache = args.out.with_suffix(".raw.json")
    cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(f"\n结果 {args.out}\n原始 {cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
