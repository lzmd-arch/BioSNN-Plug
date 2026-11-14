"""**非线性融合**：把快半的分布换成"上下文内排名"，只用顺序、不用幅度。

动机：快半的 log 概率幅度被压在 0.16 nats 内，而 `log q` 的跨度是 3–4 nats。
线性的 `log q + α·log p` 因此几乎推不动任何东西——今天测到的"唯一增益"是温度效应。
**幅度塌缩不等于没有信息：顺序可能还在。**

四个变体（同一批位置、同一次划分）：
  raw     log q + α·log p           （现行）
  temp    z(log q) 单独              （温度对照，必须被比较）
  rank    log q + α·(−排名_p)        （秩融合，非线性）
  agree   log q + α·[快半首选==慢半首选] （一致性门，非线性）
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


def align(slow, fast):
    out = {}
    for tok, _ in slow:
        m = sum(math.exp(lp) for st, lp in fast if st and tok.startswith(st))
        out[tok] = math.log(max(m, 1e-12))
    return out


def mass(table, truth):
    if not table:
        return -99.0
    v = np.array(list(table.values()), float)
    v -= v.max()
    p = np.exp(v)
    hit = sum(pi for pi, t in zip(p, table, strict=True) if t and truth.startswith(t))
    return float(math.log(max(hit / p.sum(), 1e-12)))


def top1(table):
    return max(table.items(), key=lambda kv: kv[1])[0] if table else ""


def main():
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
            if slow and fast:
                rows.append({"truth": truth, "slow": slow, "fast": fast})
        if len(rows) >= 500:
            break
    print(f"位置 {len(rows)}", flush=True)

    def evaluate(fn, alphas):
        best = None
        for a in alphas:
            ms, hs = [], []
            for r in rows:
                t = fn(r, a)
                ms.append(mass(t, r["truth"]))
                hs.append(float(bool(top1(t)) and r["truth"].startswith(top1(t))))
            row = (round(float(np.mean(ms)), 4), round(float(np.mean(hs)), 4), a)
            if best is None or row[0] > best[0]:
                best = row
        return best

    def raw(r, a):
        if a == 0:
            return dict(r["slow"])
        al = align(r["slow"], r["fast"])
        floor = min(lp for _, lp in r["slow"]) - 5.0
        t = {tok: lp + a * al.get(tok, floor) for tok, lp in r["slow"]}
        for st, slp in r["fast"]:
            if st and not any(tok.startswith(st) for tok, _ in r["slow"]):
                t.setdefault(st, floor + a * slp)
        return t

    def temp(r, a):
        q = np.array([lp for _, lp in r["slow"]])
        z = (q - q.mean()) / (q.std() + 1e-9)
        return {r["slow"][i][0]: float(z[i]) for i in range(len(q))}

    def rank(r, a):
        order = sorted(range(len(r["slow"])), key=lambda i: -r["slow"][i][1])
        _rank_of = {r["slow"][i][0]: k for k, i in enumerate(order)}
        al = align(r["slow"], r["fast"])
        fr = sorted(al.items(), key=lambda kv: -kv[1])
        frank = {tok: k for k, (tok, _) in enumerate(fr)}
        return {tok: lp - a * frank.get(tok, len(fr)) for tok, lp in r["slow"]}

    def agree(r, a):
        best = max(r["slow"], key=lambda kv: kv[1])[0]
        ok = any(best.startswith(st) or st.startswith(best) for st, _ in r["fast"] if st)
        return {tok: lp + (a if (ok and tok == best) else 0.0) for tok, lp in r["slow"]}

    out = {}
    out["raw"] = evaluate(raw, (0.0, 0.03, 0.3, 1.0))
    out["temp"] = evaluate(temp, (0.0,))
    out["rank"] = evaluate(rank, (0.1, 0.5, 1.0, 2.0))
    out["agree"] = evaluate(agree, (0.5, 1.0, 2.0))
    for k, v in out.items():
        print(f"  {k:6s} 最好：真值质量 {v[0]:+.4f}  top-1 {v[1]:.4f}  (α={v[2]})", flush=True)

    _base = out["raw"][0] if out["raw"][2] == 0.0 else None
    print("\n结论要看 rank/agree 是否高于 temp（温度对照）")
    p = Path("sweep_results/lm_fusion/nonlinear.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {"positions": len(rows), **{k: list(v) for k, v in out.items()}},
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
