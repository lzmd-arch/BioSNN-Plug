"""融合公式的**尺度**问题：`log p_fast` 的动态范围只有 `log q` 的 1/20。

实测：快半 top-10 的概率和只有 0.004–0.016，十个候选的 logprob 全挤在 0.16 nats 内；
慢半 top-20 的跨度是 3–4 nats。于是 `α=0.03` 时快半能造成的分数变化约 0.005 nats，
相对影响力 0.17% —— **"α 平台"可能不是质量平台，是"推不动"的平台**。

本脚本把两项各自标准化后再融合，重扫 α：

    raw    : score = log q + α · log p
    scaled : score = z(log q) + α · z(log p)      # 各自按候选集内的标准差标准化

判据（写死）：若 scaled 在某个 α 上的真值质量**明显高于** raw 的最好值，
则"融合没有信息"这个结论**作废**，以前的 α 扫描都在无效区。
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
FAST = "http://127.0.0.1:18763/complete"
DB = Path("D:/AxoData/memory/mem.db")
ALPHAS = (0.0, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5)


def post(url: str, payload: dict, timeout: float = 180.0) -> dict:
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


def align(slow: list[tuple[str, float]], fast: list[tuple[str, float]]) -> dict[str, float]:
    """把快半的质量按字符串前缀归并到慢半的候选上（与 `semprea_fuse.py` 同口径）。"""
    table = {}
    for token, _ in slow:
        mass = sum(math.exp(lp) for stok, lp in fast if stok and token.startswith(stok))
        table[token] = math.log(max(mass, 1e-12))
    return table


def raw_table(slow, fast, alpha: float) -> dict[str, float]:
    if not slow:
        return {}
    if not fast:
        return dict(slow)
    aligned = align(slow, fast)
    return {tok: lp + alpha * aligned.get(tok, -20.0) for tok, lp in slow}


def scaled_table(slow, fast, alpha: float) -> dict[str, float]:
    """两项各自按**候选集内的标准差**标准化，再线性组合。"""
    if not slow:
        return {}
    if alpha == 0.0 or not fast:
        return dict(slow)
    aligned = align(slow, fast)
    q = np.array([lp for _, lp in slow], dtype=np.float64)
    p = np.array([aligned.get(tok, -20.0) for tok, _ in slow], dtype=np.float64)
    zq = (q - q.mean()) / (q.std() + 1e-9)
    zp = (p - p.mean()) / (p.std() + 1e-9)
    # **α=0 时这里返回 z(q) 单独**——它是"只做温度缩放"的对照组。
    # 没有它，就分不清增益来自快半还是来自 softmax 温度。
    return {slow[i][0]: float(zq[i] + alpha * zp[i]) for i in range(len(slow))}


def truth_mass(table: dict[str, float], truth: str) -> float:
    if not table:
        return -99.0
    vals = np.array(list(table.values()), dtype=np.float64)
    vals -= vals.max()
    probs = np.exp(vals)
    total = probs.sum() or 1e-12
    hit = sum(pr for pr, tok in zip(probs, table, strict=True) if tok and truth.startswith(tok))
    return float(math.log(max(hit / total, 1e-12)))


def top1(table: dict[str, float]) -> str:
    return max(table.items(), key=lambda kv: kv[1])[0] if table else ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="融合尺度重测")
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--min-len", type=int, default=24)
    ap.add_argument("--per-entry", type=int, default=12)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--k", type=int, default=10, help="快半候选数")
    ap.add_argument("--out", type=Path, default=Path("sweep_results/lm_fusion/scaled_v1.json"))
    args = ap.parse_args(argv)

    texts = read_memories(args.db, args.min_len)
    holdout = texts[int(len(texts) * 0.6) :]
    rows = []
    for text in holdout:
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
                fast = [
                    (c["token"], float(c["logprob"]))
                    for c in post(FAST, {"context": context[-256:], "k": args.k})["candidates"]
                ]
            except Exception as exc:
                print(f"  跳过 {type(exc).__name__}", flush=True)
                continue
            if not slow or not fast:
                continue
            rows.append({"context": context, "truth": truth, "slow": slow, "fast": fast})
        if len(rows) >= 700:
            break
    print(f"位置 {len(rows)}", flush=True)

    report = {}
    for name, builder in (("raw", raw_table), ("scaled", scaled_table)):
        table_rows = []
        for alpha in ALPHAS:
            masses, hits = [], []
            for row in rows:
                table = builder(row["slow"], row["fast"], alpha)
                masses.append(truth_mass(table, row["truth"]))
                hits.append(float(bool(top1(table)) and row["truth"].startswith(top1(table))))
            table_rows.append(
                {
                    "alpha": alpha,
                    "mean_mass": round(float(np.mean(masses)), 4),
                    "top1": round(float(np.mean(hits)), 4),
                }
            )
        report[name] = table_rows
        print(f"\n[{name}]")
        for r in table_rows:
            print(f"  α={r['alpha']:<5} 真值质量 {r['mean_mass']:+.4f}  top-1 {r['top1']:.4f}")

    raw_best = max(r["mean_mass"] for r in report["raw"])
    scaled_best = max(r["mean_mass"] for r in report["scaled"])
    improvement = scaled_best - raw_best
    verdict = (
        "融合的旧结论可能建立在无效区：标准化后明显更高"
        if improvement > 0.2
        else "标准化后没有更好 ⇒ 快半在这批位置上确实没有可用信息"
    )
    print(f"\n标准化最好 {scaled_best:+.4f} vs 原式最好 {raw_best:+.4f}，差 {improvement:+.4f}")
    print(verdict)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "positions": len(rows),
                "report": report,
                "improvement": round(improvement, 4),
                "verdict": verdict,
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\n结果 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
