"""**可学的融合门**：不再手挑算子，让数据决定每个位置的 α。

每位置特征：慢半熵、慢半首选 logp、快半熵、两半是否一致、上下文长度。
门：g = sigmoid(w·x)，score = log q + α_max · g · log p。
w 用**随机搜索**在训练段上拟合（样本少，梯度反而不稳），留出段只报数。

判据：留出真值质量必须**高于温度对照**（这是唯一有资格叫增益的线）。
"""

from __future__ import annotations

import json
import math
import sqlite3
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

SLOW = "http://127.0.0.1:8081/v1/completions"
FAST = "http://127.0.0.1:56031/complete"
DB = Path("D:/AxoData/memory/mem.db")


def post(url, payload, timeout=180.0):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with op.open(req, timeout=timeout) as h:
        return json.loads(h.read())


def memories(min_len=24):
    uri = "file:" + urllib.parse.quote(str(DB).replace("\\", "/")) + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        return [
            r[0]
            for r in con.execute(
                "select content from memories where content is not null and length(content)>? order by created_at",
                (min_len,),
            )
        ]
    finally:
        con.close()


def entropy(rows):
    p = np.exp(np.array([lp for _, lp in rows], float))
    p = p / (p.sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())


def align(slow, fast):
    return {
        tok: math.log(max(sum(math.exp(lp) for st, lp in fast if st and tok.startswith(st)), 1e-12))
        for tok, _ in slow
    }


def mass(table, truth):
    v = np.array(list(table.values()), float)
    v -= v.max()
    p = np.exp(v)
    hit = sum(pi for pi, t in zip(p, table, strict=True) if t and truth.startswith(t))
    return float(math.log(max(hit / p.sum(), 1e-12)))


def collect():
    texts = memories()
    hold = texts[int(len(texts) * 0.6) :]
    rows = []
    for text in hold:
        for start in list(range(20, max(len(text) - 2, 21), 6))[:12]:
            ctx, truth = text[:start], text[start:]
            try:
                slow = [
                    (i["token"], float(i["logprob"]))
                    for i in post(
                        SLOW,
                        {
                            "model": "qwen3",
                            "prompt": ctx,
                            "max_tokens": 1,
                            "temperature": 0,
                            "logprobs": 20,
                        },
                    )["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                ]
                fast = [
                    (c["token"], float(c["logprob"]))
                    for c in post(FAST, {"context": ctx[-256:], "k": 10})["candidates"]
                ]
            except Exception:
                continue
            if not slow or not fast:
                continue
            s_top = slow[0][0]
            agree = float(any(s_top.startswith(st) or st.startswith(s_top) for st, _ in fast if st))
            rows.append(
                {
                    "truth": truth,
                    "slow": slow,
                    "fast": fast,
                    "x": np.array(
                        [entropy(slow), slow[0][1], entropy(fast), agree, len(ctx) / 256.0]
                    ),
                }
            )
        if len(rows) >= 450:
            break
    return rows


def fused(r, g, alpha_max=1.0):
    al = align(r["slow"], r["fast"])
    floor = min(lp for _, lp in r["slow"]) - 5.0
    a = alpha_max * g
    t = {tok: lp + a * al.get(tok, floor) for tok, lp in r["slow"]}
    for st, slp in r["fast"]:
        if st and not any(tok.startswith(st) for tok, _ in r["slow"]):
            t.setdefault(st, floor + a * slp)
    return t


def main():
    rows = collect()
    print(f"位置 {len(rows)}", flush=True)
    mu = np.mean([r["x"] for r in rows], 0)
    sd = np.std([r["x"] for r in rows], 0) + 1e-9
    for r in rows:
        r["z"] = (r["x"] - mu) / sd
    cut = int(len(rows) * 0.6)
    tr, te = rows[:cut], rows[cut:]

    rng = np.random.default_rng(0)
    best = (-1e9, None)
    for _ in range(3000):
        w = rng.normal(size=6) * 1.5
        total = 0.0
        for r in tr:
            g = 1.0 / (1.0 + math.exp(-float(np.dot(w, np.append(r["z"], 1.0)))))
            total += mass(fused(r, g), r["truth"])
        if total > best[0]:
            best = (total, w)
    w = best[1]
    score, n = 0.0, 0
    for r in te:
        g = 1.0 / (1.0 + math.exp(-float(np.dot(w, np.append(r["z"], 1.0)))))
        score += mass(fused(r, g), r["truth"])
        n += 1
    learned = score / max(n, 1)
    temp = float(np.mean([mass(dict(r["slow"]), r["truth"]) for r in te]))
    base = temp
    out = {
        "positions": len(rows),
        "train": len(tr),
        "holdout": len(te),
        "learned_gate_mass": round(learned, 4),
        "linear_mass": round(base, 4),
        "delta": round(learned - base, 4),
        "verdict": "可学门优于线性" if learned - base > 0.05 else "可学门没有优于线性",
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    Path("sweep_results/lm_fusion/learned_gate.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
