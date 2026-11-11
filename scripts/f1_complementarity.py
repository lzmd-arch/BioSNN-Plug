"""融合之前必须先问：**SNN 有没有在千问答错的地方答对？**

如果没有，`log q + α·log p_snn` 只会把正确的分布往错的方向拉——融合在结构上就是死的，
不管 α 怎么调。

判据 `F-0`：在留出文本上，**"千问 top-1 错 且 SNN top-1 对"的比例 ≥ 5%**。
低于这个数，融合没有可利用的互补性，L3 直接不做。

（用字符串前缀判断对错：两个模型的分词器不同，只有解码后的文本可比。）
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SNN = "http://127.0.0.1:18735/complete"
OLLAMA = "http://127.0.0.1:11434/api/generate"
CORPUS = Path("D:/AxoData/sel/lm/corpus/skypile/data")


def post(url: str, payload: dict, timeout: float = 60.0) -> dict:
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F-0：两模型的互补性")
    ap.add_argument("--positions", type=int, default=40)
    ap.add_argument("--context-chars", type=int, default=24)
    ap.add_argument("--shard", default="2021-04_zh_head_0001.jsonl", help="训练集之外的分片")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    path = CORPUS / args.shard
    rng = random.Random(args.seed)
    pool = []
    for _, text in zip(range(400), texts(path), strict=False):
        pool.append(text)
    rng.shuffle(pool)
    rows = []
    for text in pool:
        if len(rows) >= args.positions:
            break
        start = rng.randrange(0, max(len(text) - args.context_chars - 4, 1))
        context = text[start : start + args.context_chars]
        truth = text[start + args.context_chars : start + args.context_chars + 4]
        if not truth.strip():
            continue
        try:
            snn = post(SNN, {"context": context, "k": 5}, timeout=30)
            qwen = post(
                OLLAMA,
                {
                    "model": "qwen3-4b",
                    "prompt": context,
                    "stream": False,
                    "raw": True,
                    "options": {"num_predict": 1, "temperature": 0},
                    "logprobs": True,
                    "top_logprobs": 5,
                },
                timeout=90,
            )
        except Exception as exc:
            print(f"  跳过：{type(exc).__name__}", flush=True)
            continue
        snn_tokens = [c["token"] for c in snn.get("candidates", [])]
        logprobs = qwen.get("logprobs") or []
        qwen_tokens = [x["token"] for x in (logprobs[0]["top_logprobs"] if logprobs else [])]
        hit_snn = bool(snn_tokens) and truth.startswith(snn_tokens[0])
        hit_qwen = bool(qwen_tokens) and truth.startswith(qwen_tokens[0])
        # 前缀重叠：SNN 的首选是不是 truth 的前缀（分词器不同，只能这样比）
        rows.append(
            {
                "context": context[-12:],
                "truth": truth[:2],
                "snn": snn_tokens[:2],
                "qwen": qwen_tokens[:2],
                "hit_snn": hit_snn,
                "hit_qwen": hit_qwen,
            }
        )

    n = len(rows)
    both = sum(1 for r in rows if r["hit_snn"] and r["hit_qwen"])
    only_q = sum(1 for r in rows if r["hit_qwen"] and not r["hit_snn"])
    only_s = sum(1 for r in rows if r["hit_snn"] and not r["hit_qwen"])
    neither = sum(1 for r in rows if not r["hit_snn"] and not r["hit_qwen"])
    print(f"\nn={n}")
    print(f"  两个都对 {both}   只千问对 {only_q}   **只 SNN 对 {only_s}**   都不对 {neither}")
    print(f"  千问 top-1 准确率 {sum(1 for r in rows if r['hit_qwen']) / max(n, 1):.3f}")
    print(f"  SNN  top-1 准确率 {sum(1 for r in rows if r['hit_snn']) / max(n, 1):.3f}")
    share = only_s / max(n, 1)
    print(f"\n判据 F-0（只 SNN 对 ≥5%）：{share:.3%} ⇒ {'过' if share >= 0.05 else '不过'}")
    for row in rows[:6]:
        print(f"  {row['context']}｜真值 {row['truth']!r} SNN {row['snn']} Qwen {row['qwen']}")
    out = Path("sweep_results/lm_fusion/f0_complementarity.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"n": n, "rows": rows}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
