"""SNN 的特长能不能兑现：**让状态读完整条流，而不是读一张 256 字的裁剪**。

快半是逐时刻递推的，状态可以一直往后接（`serve_snn` 的 `session`）。
transformer 要同样的历史就得 KV cache 线性增长；SNN 这边是 O(1)/token。

问题是：**那份状态里还有没有信息？** 这是可测的，不是可假设的。

协议（先写死）：

1. 只取**长记忆**（> 1200 字），因为短记忆上"全量"与"裁剪 256"没有区别；
2. 位置：从第 400 字起、步长 400，最多到 8000 字；
3. 每个位置两种取法，**同一位置配对**：
   - `crop`：`context[-256:]`，不带 session（现行做法）；
   - `stream`：**完整 context**，带 session、`max_len=8192`（状态从头接到这里）；
4. 判据 **S-1**：留出记忆上 `stream` 的 top-1 命中率 **高于** `crop`，
   且按记忆聚类的 bootstrap 95% 区间**下界 > 0**；
5. 同时记录每次调用的耗时——用来验证"增量是 O(1)"这个说法。

**若 S-1 不过**：写"状态在 256 字之后不再提供可测的信息"，并把这条记成
"SNN 的长历史优势在**这个模型上**没有兑现"，而不是"SNN 没有这个优势"。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

SLOW = "http://127.0.0.1:8081/v1/completions"
FAST = "http://127.0.0.1:18762/complete"
DB = Path("D:/AxoData/memory/mem.db")


def post(url: str, payload: dict, timeout: float = 600.0) -> dict:
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


def topk(cands, truth: str, k: int) -> bool:
    return any(hits(t, truth) for t, _ in cands[:k])


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="长记忆流式读取（SNN 特长）")
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--min-len", type=int, default=1200)
    ap.add_argument("--start", type=int, default=400)
    ap.add_argument("--step", type=int, default=400)
    ap.add_argument("--max-context", type=int, default=8000)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("sweep_results/lm_fusion/longmem_v1.json"))
    args = ap.parse_args(argv)

    texts = read_memories(args.db, args.min_len)
    cut = int(len(texts) * args.train_frac)
    splits = {"train": texts[:cut], "holdout": texts[cut:]}
    print(f"长记忆 {len(texts)} 条：训练 {cut} / 留出 {len(texts) - cut}", flush=True)

    rows: list[dict] = []
    for split, entries in splits.items():
        for entry_id, text in enumerate(entries):
            session = f"stream-{split}-{entry_id}"
            limit = min(len(text) - 2, args.max_context)
            for start in range(args.start, limit, args.step):
                context, truth = text[:start], text[start:]
                t0 = time.perf_counter()
                crop = [
                    (c["token"], float(c["logprob"]))
                    for c in post(FAST, {"context": context[-256:], "k": 10})["candidates"]
                ]
                t_crop = time.perf_counter() - t0
                t0 = time.perf_counter()
                stream = [
                    (c["token"], float(c["logprob"]))
                    for c in post(
                        FAST,
                        {
                            "context": context,
                            "k": 10,
                            "session": session,
                            "max_len": args.max_context,
                        },
                    )["candidates"]
                ]
                t_stream = time.perf_counter() - t0
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
                rows.append(
                    {
                        "split": split,
                        "entry": f"{split}-{entry_id}",
                        "context_len": len(context),
                        "truth": truth[:8],
                        "slow1": slow[0][0],
                        "crop1": crop[0][0],
                        "stream1": stream[0][0],
                        "hit_slow": int(hits(slow[0][0], truth)),
                        "hit_crop_top1": int(hits(crop[0][0], truth)),
                        "hit_stream_top1": int(hits(stream[0][0], truth)),
                        "hit_crop_top5": int(topk(crop, truth, 5)),
                        "hit_stream_top5": int(topk(stream, truth, 5)),
                        "t_crop": round(t_crop, 3),
                        "t_stream": round(t_stream, 3),
                    }
                )
                print(
                    f"  [{split}] ctx {len(context):>5}  "
                    f"crop {rows[-1]['hit_crop_top1']}  stream {rows[-1]['hit_stream_top1']}  "
                    f"慢半 {rows[-1]['hit_slow']}  ({t_crop:.2f}s / {t_stream:.2f}s)",
                    flush=True,
                )

    hold = [r for r in rows if r["split"] == "holdout"]
    if not hold:
        print("留出集为空，作废")
        return 1
    groups = [r["entry"] for r in hold]
    crop = np.array([r["hit_crop_top1"] for r in hold], dtype=float)
    stream = np.array([r["hit_stream_top1"] for r in hold], dtype=float)
    slow = np.array([r["hit_slow"] for r in hold], dtype=float)
    lo, hi = cluster_ci(stream - crop, groups)
    summary = {
        "holdout_positions": len(hold),
        "memories": {"train": len(splits["train"]), "holdout": len(splits["holdout"])},
        "context_len": {
            "min": min(r["context_len"] for r in hold),
            "max": max(r["context_len"] for r in hold),
            "mean": round(float(np.mean([r["context_len"] for r in hold])), 1),
        },
        "top1": {
            "slow": round(float(slow.mean()), 4),
            "crop256": round(float(crop.mean()), 4),
            "stream_full": round(float(stream.mean()), 4),
            "stream_minus_crop_pp": round(float((stream - crop).mean()) * 100, 2),
            "stream_minus_crop_ci95_pp": [round(lo * 100, 2), round(hi * 100, 2)],
        },
        "top5": {
            "crop256": round(float(np.mean([r["hit_crop_top5"] for r in hold])), 4),
            "stream_full": round(float(np.mean([r["hit_stream_top5"] for r in hold])), 4),
        },
        "latency_s": {
            "crop_mean": round(float(np.mean([r["t_crop"] for r in rows])), 4),
            "stream_mean": round(float(np.mean([r["t_stream"] for r in rows])), 4),
        },
        "verdict": (
            "S-1 过：状态在 256 字之后仍提供信息"
            if lo > 0
            else "S-1 不过：状态在 256 字之后没有可测增益"
        ),
    }
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    raw = args.out.with_suffix(".raw.json")
    raw.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(f"\n结果 {args.out}\n原始 {raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
