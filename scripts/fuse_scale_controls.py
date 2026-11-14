"""融合尺度：把该有的对照一次补齐（同一批位置、同一份数据）。

四个变体：

1. `raw`        : log q + α·log p                （现行公式）
2. `temp`       : z(log q) 单独                  （**温度对照**——若它已经拿到全部增益，
                                                  则增益来自 softmax 温度，与快半无关）
3. `zscore`     : z(log q) + α·z(log p)
4. `logsoftmax` : logsoftmax_q + α·logsoftmax_p  （按候选集真正归一化，不做线性拉伸）

另加 `pmi` : z(log q) + α·z(log p − log prior)，检验去偏在**正确尺度**下是否仍无效。

判据（写死）：只有 `zscore`/`logsoftmax`/`pmi` 显著**优于 `temp`** 才算快半提供了信息；
只优于 `raw` 不够——那可能只是温度。
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
PRIOR = Path("sweep_results/lm_semprea_v1/counts_32000.npy")
ALPHAS = (0.0, 0.01, 0.05, 0.1, 0.3)


def post(url: str, payload: dict, timeout: float = 180.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as handle:
        return json.loads(handle.read())


def memories(db: Path, min_len: int) -> list[str]:
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
    return [r[0] for r in rows]


def zscore(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / (values.std() + 1e-9)


def log_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    return shifted - math.log(np.exp(shifted).sum())


def truth_mass(table: dict[str, float], truth: str) -> float:
    if not table:
        return -99.0
    vals = np.array(list(table.values()), dtype=np.float64)
    vals = vals - vals.max()
    probs = np.exp(vals)
    total = probs.sum() or 1e-12
    hit = sum(p for p, tok in zip(probs, table, strict=True) if tok and truth.startswith(tok))
    return float(math.log(max(hit / total, 1e-12)))


def top1(table: dict[str, float]) -> str:
    return max(table.items(), key=lambda kv: kv[1])[0] if table else ""


def build(row, alpha: float, kind: str, priors: dict[str, float]):
    slow, fast = row["slow"], row["fast"]
    if not slow:
        return {}
    toks = [t for t, _ in slow]
    q = np.array([lp for _, lp in slow], dtype=np.float64)
    if kind == "raw":
        if alpha == 0.0 or not fast:
            return dict(slow)
        aligned = np.array(
            [
                math.log(
                    max(sum(math.exp(lp) for st, lp in fast if st and tok.startswith(st)), 1e-12)
                )
                for tok in toks
            ]
        )
        return {toks[i]: float(q[i] + alpha * aligned[i]) for i in range(len(toks))}
    if not fast:
        return {toks[i]: float(q[i]) for i in range(len(toks))}
    p = np.array(
        [
            math.log(max(sum(math.exp(lp) for st, lp in fast if st and tok.startswith(st)), 1e-12))
            for tok in toks
        ]
    )
    if kind == "pmi":
        prior = np.array([priors.get(tok, 0.0) for tok in toks])
        p = p - prior
    if kind == "temp":
        return {toks[i]: float(zscore(q)[i]) for i in range(len(toks))}
    if kind == "zscore":
        return {toks[i]: float(zscore(q)[i] + alpha * zscore(p)[i]) for i in range(len(toks))}
    # logsoftmax
    return {toks[i]: float(log_softmax(q)[i] + alpha * log_softmax(p)[i]) for i in range(len(toks))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="融合尺度：补齐对照")
    ap.add_argument("--min-len", type=int, default=24)
    ap.add_argument("--per-entry", type=int, default=12)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("sweep_results/lm_fusion/scale_controls.json"))
    args = ap.parse_args(argv)

    texts = memories(DB, args.min_len)
    holdout = texts[int(len(texts) * 0.6) :]
    priors: dict[str, float] = {}
    if PRIOR.is_file():
        counts = np.load(PRIOR).astype(np.float64)
        total = max(counts.sum(), 1.0)
        import sys

        sys.path.insert(0, "D:/Projects/BioSNN-Plug")
        from research.sel.lm.subset_tokenizer import SubsetTokenizer

        bridge = SubsetTokenizer(Path("D:/AxoData/sel/lm/corpus/qwen_tok_fw2"))
        for i in range(len(counts)):
            text = bridge.decode([i])
            if text:
                priors[text] = math.log(max(counts[i] / total, 1e-12))

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
            except Exception:
                continue
            if slow and fast:
                rows.append({"truth": truth, "slow": slow, "fast": fast})
        if len(rows) >= 700:
            break
    print(f"位置 {len(rows)}", flush=True)

    report: dict = {}
    for kind in ("raw", "temp", "zscore", "logsoftmax", "pmi"):
        alphas = (0.0,) if kind == "temp" else ALPHAS
        for alpha in alphas:
            masses, hits = [], []
            for row in rows:
                table = build(row, alpha, kind, priors)
                masses.append(truth_mass(table, row["truth"]))
                hits.append(float(bool(top1(table)) and row["truth"].startswith(top1(table))))
            key = f"{kind}@{alpha}"
            report[key] = {
                "mass": round(float(np.mean(masses)), 4),
                "top1": round(float(np.mean(hits)), 4),
            }
            print(
                f"  {key:18s} 真值质量 {report[key]['mass']:+.4f}  top-1 {report[key]['top1']:.4f}",
                flush=True,
            )

    base = report["raw@0.0"]["mass"]
    temp = report["temp@0.0"]["mass"]
    best_scaled = max(
        v["mass"] for k, v in report.items() if k.startswith(("zscore", "logsoftmax", "pmi"))
    )
    print(f"\n基线 raw@0 {base:+.4f}；温度对照 temp {temp:+.4f}（{temp - base:+.4f}）")
    print(f"最好的一次融合 {best_scaled:+.4f}，相对温度对照 {best_scaled - temp:+.4f}")
    verdict = (
        "快半提供了信息：融合优于温度对照"
        if best_scaled - temp > 0.02
        else "**+0.18 全是 softmax 温度**：融合相对温度对照没有增益 ⇒ 快半仍然没有可用信息"
    )
    print(verdict)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"positions": len(rows), "report": report, "verdict": verdict},
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
