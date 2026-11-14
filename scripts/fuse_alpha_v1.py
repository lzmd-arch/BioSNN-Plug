"""用 **V1 的快半**重跑 α 扫描：融合到底有没有用？

历史结论（α\\* = 0、剂量–反应）是在**旧的快半**上测的（SkyPile 语料、旧子集、旧树）。
V1 换了语料、词表、树，top-1 也高了一截，**必须重测**。

做法与 U-2 同口径（本机、只用聚合指标、不打印正文）：

1. 从真库按时间取记忆文本，后 20% 留出；
2. 逐个位置取前缀（前 80 字），真值 = 后面的内容；
3. 慢半（llama-server `/v1/completions`，logprobs + top_logprobs）给 top-20；
4. 快半（`serve_snn` `/complete`）给 top-10；
5. 在**并集**上算 `log q + α·log p`，比真值前缀的概率质量。

**前缀归并**：慢半的 token "同时" 收到的快半质量 = 我方所有是它前缀的 token 概率之和。
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import urllib.parse
import urllib.request
from pathlib import Path

SLOW = "http://127.0.0.1:8081/v1/completions"
FAST = "http://127.0.0.1:18734/complete"
DB = Path("D:/AxoData/memory/mem.db")


def post(url: str, payload: dict, timeout: float = 120.0) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    # **loopback 必须绕开系统代理**（本仓踩过：代理会把 127.0.0.1 送进 Clash）
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as handle:
        return json.loads(handle.read())


def read_memories(db: Path) -> list[str]:
    uri = "file:" + urllib.parse.quote(str(db).replace("\\", "/")) + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = con.execute(
            "select content from memories where content is not null and length(content) > 8 "
            "order by created_at"
        ).fetchall()
    finally:
        con.close()
    return [row[0] for row in rows]


def slow_logprobs(context: str, top_n: int = 20) -> list[tuple[str, float]]:
    payload = {
        "model": "qwen3",
        "prompt": context,
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": top_n,
    }
    body = post(SLOW, payload)
    content = body["choices"][0]["logprobs"]["content"][0]
    return [(item["token"], float(item["logprob"])) for item in content["top_logprobs"]]


def fast_logprobs(context: str, k: int = 10) -> list[tuple[str, float]]:
    body = post(FAST, {"context": context[-256:], "k": k})
    return [(c["token"], float(c["logprob"])) for c in body.get("candidates") or []]


def mass(scores: dict[str, float], truth: str) -> float:
    """真值前缀的概率质量（与快慢两半同口径）。"""
    total = sum(math.exp(v) for v in scores.values()) or 1e-12
    hit = sum(math.exp(v) for t, v in scores.items() if t and truth.startswith(t))
    return math.log(max(hit / total, 1e-12))


def fuse(slow, fast, alpha: float, floor_gap: float = 5.0) -> dict[str, float]:
    if not slow:
        return {}
    if not fast or alpha == 0.0:
        return dict(slow)
    floor = min(lp for _, lp in slow) - floor_gap
    table: dict[str, float] = {}
    for token, logprob in slow:
        m = sum(math.exp(slp) for stoken, slp in fast if stoken and token.startswith(stoken))
        table[token] = logprob + alpha * math.log(max(m, 1e-12))
    for stoken, slp in fast:
        if stoken and not any(token.startswith(stoken) for token, _ in slow):
            table.setdefault(stoken, floor + alpha * slp)
    return table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="V1 快半的 α 扫描")
    ap.add_argument("--positions", type=int, default=60)
    ap.add_argument("--max-holdout", type=int, default=40)
    ap.add_argument("--prefix", type=int, default=80)
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--out", type=Path, default=Path("sweep_results/lm_fusion/alpha_v1.json"))
    args = ap.parse_args(argv)

    texts = read_memories(args.db)
    holdout = texts[-args.max_holdout :] if texts else []
    print(f"真库 {len(texts)} 条，留出 {len(holdout)} 条", flush=True)

    alphas = [0.0, 0.05, 0.1, 0.25, 0.5]
    totals = dict.fromkeys(alphas, 0.0)
    top1 = dict.fromkeys(alphas, 0)
    counted = 0
    raw: list[dict] = []
    for text in holdout:
        step = max(len(text) // max(args.positions // max(len(holdout), 1), 1), 1)
        for start in range(8, len(text) - 2, max(step, 4)):
            if counted >= args.positions:
                break
            context, truth = text[:start], text[start:]
            try:
                slow = slow_logprobs(context)
                fast = fast_logprobs(context)
            except Exception as exc:  # 端点抖动就跳过这一点，不填数
                print(f"  跳过：{type(exc).__name__}", flush=True)
                continue
            if not slow:
                continue
            counted += 1
            raw.append({"context": context, "truth": truth, "slow": slow, "fast": fast})
            for alpha in alphas:
                table = fuse(slow, fast, alpha)
                if not table:
                    continue
                totals[alpha] += mass(table, truth)
                best = max(table.items(), key=lambda kv: kv[1])[0]
                top1[alpha] += int(bool(best) and truth.startswith(best))
        if counted >= args.positions:
            break

    n = max(counted, 1)
    rows = [
        {
            "alpha": a,
            "mean_logp": round(totals[a] / n, 4),
            "top1": round(top1[a] / n, 4),
        }
        for a in alphas
    ]
    print(f"\n位置数 {counted}")
    for row in rows:
        print(
            f"  α={row['alpha']:<5} 真值平均 log p {row['mean_logp']:+.4f}  top-1 {row['top1']:.4f}"
        )

    base = rows[0]["mean_logp"]
    best = max(rows, key=lambda r: r["mean_logp"])
    print(f"\n最优 α = {best['alpha']}，相对 α=0 的改善 {best['mean_logp'] - base:+.4f}")

    cache = args.out.with_suffix(".raw.json")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(f"原始响应缓存 {cache}（{len(raw)} 条）")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"positions": counted, "rows": rows, "best_alpha": best["alpha"]},
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
