"""U-2：用户头在这个人**自己的**文本上到底有没有预测力。

做法（全部本地、只出聚合指标，**不打印任何正文**）：

1. 从真库按时间读出全部记忆文本，**前 80% 训练、后 20% 留出**；
2. 用户头 = 这批 token 的指数遗忘计数（`axo.llm.user_head`）；
3. 在留出段上逐个位置问"底座模型"（默认本地 SNN 服务）要 top-k，
   然后比 `score = log q + γ·log u` 在 **γ=0 与 γ>0** 下给真值的概率质量。

判据 **U-2**：存在 γ>0 使留出段的平均 log p **高于** γ=0。
不过 ⇒ 用户头对这个人的下一个字符没有预测力，收口。
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

SNN = "http://127.0.0.1:18735/complete"
BPE = Path("D:/AxoData/sel/lm/corpus/skypile_tok/bpe")
DB = Path("D:/AxoData/memory/mem.db")


def post(url: str, payload: dict, timeout: float = 30.0) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as handle:
        return json.loads(handle.read())


def read_memories(db: Path) -> list[str]:
    uri = "file:" + urllib.parse.quote(str(db).replace("\\", "/")) + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = con.execute(
            "select content from memories where content is not null and length(content) > 8"
            " order by coalesce(created_at, rowid)"
        ).fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="U-2：用户头在真库上的预测力")
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--bpe", type=Path, default=BPE)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--positions", type=int, default=20, help="每条留出文本取多少个位置")
    ap.add_argument("--max-holdout", type=int, default=40)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--gammas", default="0,0.5,1,2")
    ap.add_argument("--out", type=Path, default=Path("sweep_results/lm_fusion/u2_user_head.json"))
    args = ap.parse_args(argv)

    import sys

    sys.path.insert(0, str(Path("D:/Projects/BioSNN-MemoryTerminal/src")))
    from axo.llm.user_head import UserHead
    from tokenizers import ByteLevelBPETokenizer

    texts = read_memories(args.db)
    if len(texts) < 20:
        print(f"真库太小：{len(texts)} 条")
        return 1
    cut = int(len(texts) * (1 - args.holdout))
    train, hold = texts[:cut], texts[cut:][: args.max_holdout]

    tokenizer = ByteLevelBPETokenizer(str(args.bpe / "vocab.json"), str(args.bpe / "merges.txt"))
    vocab = json.loads((args.bpe / "vocab.json").read_text(encoding="utf-8"))
    head = UserHead.cold_start(vocab)
    ids: list[int] = []
    for text in train:
        ids.extend(tokenizer.encode(text).ids)
    head.update(ids)
    print(f"真库 {len(texts)} 条：训练 {len(train)} 条 / {len(ids)} token，留出 {len(hold)} 条")

    gammas = [float(x) for x in args.gammas.split(",") if x.strip()]
    totals = dict.fromkeys(gammas, 0.0)
    #: **逐位置**存一份：只报均值的话，60 个位置上"差 0.075 nats"是不是信号就永远说不清。
    #: 第一版只存了均值，于是 U-2 的排序显著性只能写"未建立"。
    per_position: dict[float, list[float]] = {g: [] for g in gammas}
    counted = 0
    for index, text in enumerate(hold, 1):
        step = max(len(text) // args.positions, 1)
        for start in range(4, len(text) - 2, step):
            context = text[:start]
            truth = text[start:]
            try:
                slow = post(SNN, {"context": context[-256:], "k": args.k})
            except Exception:
                continue
            candidates = [(c["token"], float(c["logprob"])) for c in slow.get("candidates") or []]
            if not candidates:
                continue
            counted += 1
            for gamma in gammas:
                scores = {t: lp + gamma * head.bias_for(t) for t, lp in candidates}
                total = sum(math.exp(v) for v in scores.values()) or 1e-12
                mass = sum(math.exp(v) for t, v in scores.items() if t and truth.startswith(t))
                value = math.log(max(mass / total, 1e-12))
                totals[gamma] += value
                per_position[gamma].append(value)
        if index % 10 == 0:
            print(f"  留出 {index}/{len(hold)}，累计位置 {counted}", flush=True)

    n = max(counted, 1)
    rows = [
        {"gamma": g, "mean_logp": round(totals[g] / n, 4), "logp": per_position[g]} for g in gammas
    ]
    print(f"\n位置数 {counted}")
    for row in rows:
        print(f"  γ={row['gamma']:<4} 真值平均 log p {row['mean_logp']:+.4f}")
    base = rows[0]["mean_logp"]
    best = max(rows, key=lambda r: r["mean_logp"])
    delta = best["mean_logp"] - base
    base_values = per_position[rows[0]["gamma"]]
    best_values = per_position[best["gamma"]]
    paired = [b - a for a, b in zip(base_values, best_values, strict=True)]
    se = float(np.std(paired, ddof=1) / math.sqrt(len(paired))) if len(paired) > 1 else 0.0
    print(
        f"\n判据 U-2：最优 γ={best['gamma']} ⇒ **{delta:+.4f}**（{'过' if delta > 0 else '不过'}；配对标准误 ≈ {se:.4f}）"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"positions": counted, "rows": rows, "delta": round(delta, 4)},
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
