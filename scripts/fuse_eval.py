"""把两个模型的分布**真的合起来**，扫 α，看融合能不能选对下一个 token。

机制（这是"组合模型"的实质）：

```
score(token) = log q_qwen(token) + α · log p_snn(token)
```

- 两边词表不同（我们 BPE-32k / 千问 152k），所以按**字符串**归并：
  千问的候选 "同时"，把我们这边所有**是它前缀**的 token 概率加起来当作 p_snn("同时")；
- 两边候选的并集上做 softmax，argmax 就是融合模型的输出——**同一次采样、同一个分布**。

判据 `F-1`：存在 α>0 使得 top-1 准确率 **高于 α=0**（=千问单独）。高多少都照报。

用法::

    uv run python scripts/fuse_eval.py --positions 60 --alphas 0,0.1,0.25,0.5,1.0
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

SNN = "http://127.0.0.1:18735/complete"
OLLAMA = "http://127.0.0.1:11434/api/generate"
CORPUS = Path("D:/AxoData/sel/lm/corpus/skypile/data")


def post(url: str, payload: dict, timeout: float = 90.0) -> dict:
    import urllib.request

    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as handle:
        return json.loads(handle.read())


def texts(path: Path):
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text") or row.get("content") or ""
            if len(text) > 60:
                yield text


def fuse(qwen: list[tuple[str, float]], snn: list[tuple[str, float]], alpha: float):
    """返回融合后的 (token, score) 列表。

    `p_snn(t)` = 我们这边所有**是 t 前缀**的 token 的概率和——跨分词器的质量标准搬法。
    """
    floor = min([lp for _, lp in qwen], default=-20.0) - 5.0
    table: dict[str, float] = {}
    for token, lp in qwen:
        table[token] = lp
    for qtoken, lq in list(table.items()):
        mass = 0.0
        for stoken, slp in snn:
            if qtoken.startswith(stoken) and stoken:
                mass += math.exp(slp)
        table[qtoken] = lq + alpha * math.log(max(mass, 1e-12))
    for stoken, slp in snn:
        if stoken and not any(q.startswith(stoken) for q, _ in qwen):
            # 千问那边没有这个候选 ⇒ 它的 log q 取观测下界（不是 -inf，否则永远选不到）
            table.setdefault(stoken, floor + alpha * slp)
    return sorted(table.items(), key=lambda kv: -kv[1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F-1：跨分词器的分布融合，扫 α")
    ap.add_argument("--positions", type=int, default=60)
    ap.add_argument("--context-chars", type=int, default=24)
    ap.add_argument("--shard", default="2021-04_zh_head_0001.jsonl")
    ap.add_argument("--alphas", default="0,0.1,0.25,0.5,1.0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    rng = random.Random(args.seed)
    pool = []
    for _, text in zip(range(500), texts(CORPUS / args.shard), strict=False):
        pool.append(text)
    rng.shuffle(pool)

    samples = []
    for text in pool:
        if len(samples) >= args.positions:
            break
        start = rng.randrange(0, max(len(text) - args.context_chars - 4, 1))
        context = text[start : start + args.context_chars]
        truth = text[start + args.context_chars : start + args.context_chars + 4]
        if not truth.strip():
            continue
        try:
            snn = post(SNN, {"context": context, "k": 10}, timeout=30)
            qwen = post(
                OLLAMA,
                {
                    "model": "qwen3-4b",
                    "prompt": context,
                    "stream": False,
                    "raw": True,
                    "options": {"num_predict": 1, "temperature": 0},
                    "logprobs": True,
                    "top_logprobs": 20,
                },
            )
        except Exception as exc:
            print(f"  跳过 {type(exc).__name__}", flush=True)
            continue
        q = [(x["token"], float(x["logprob"])) for x in qwen["logprobs"][0]["top_logprobs"]]
        s = [(c["token"], float(c["logprob"])) for c in snn.get("candidates", [])]
        if q:
            samples.append({"context": context, "truth": truth, "qwen": q, "snn": s})
        if len(samples) % 10 == 0:
            print(f"  采到 {len(samples)} 个位置", flush=True)

    n = len(samples)
    if not n:
        print("没采到样本")
        return 1
    print(f"\nn={n}，扫 α：")
    rows = []
    for alpha in alphas:
        hits = 0
        logp_truth = 0.0
        for sample in samples:
            ranked = fuse(sample["qwen"], sample["snn"], alpha)
            top = ranked[0][0] if ranked else ""
            hits += int(bool(top) and sample["truth"].startswith(top))
            # **比 top-1 更灵敏的指标**：融合分布给真值的概率质量。
            # 真值可能被切成多个 token，所以取"所有是 truth 前缀的候选"的质量和。
            total = sum(math.exp(score) for _, score in ranked) or 1e-12
            mass = sum(
                math.exp(score)
                for token, score in ranked
                if token and sample["truth"].startswith(token)
            )
            logp_truth += math.log(max(mass / total, 1e-12))
        acc = hits / n
        rows.append(
            {
                "alpha": alpha,
                "top1": round(acc, 4),
                "hits": hits,
                "mean_logp_truth": round(logp_truth / n, 4),
            }
        )
        print(
            f"  α={alpha:<5} top-1 {acc:.4f}（{hits}/{n}）  真值平均 log p {logp_truth / n:+.4f}",
            flush=True,
        )

    best = max(rows, key=lambda r: r["top1"])
    base = rows[0]["top1"]
    print(
        f"\n判据 F-1：最优 α={best['alpha']} 的 top-1 {best['top1']:.4f}，"
        f"比 α=0（千问单独）{base:.4f} **{best['top1'] - base:+.4f}**"
    )
    print("（>0 即融合有可测量的增益；=0 表示机制在但今天没有收益）", flush=True)
    out = Path("sweep_results/lm_fusion/f1_fuse.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"n": n, "rows": rows}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
