"""n-best 重排序：**让快半给整句打分，而不是只看下一个 token。**

现在的融合是逐 token 分布相加——两半答同一道题，弱的那个只是给强的加噪声，
十条改法全部失败。

这条机制不同：

    1. 让千问采样出 K 条短续写（temperature > 0，每条带自己的 logprob 之和）
    2. 让快半对**每条整句**做 teacher forcing 打分
    3. 融合分 = slow_sum + α · fast_sum，取最好的那条

小模型在**序列级重排序**上常常能提升大模型，因为它评的是一致性而不是边际。
而且它不依赖"千问有多自信"——我们已知快半的价值落在"千问自信地错"的区域。
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

SLOW = "http://127.0.0.1:8081/v1/completions"
FAST_SCORE = "http://127.0.0.1:18734/score"


def post(url: str, payload: dict, timeout: float = 120.0) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as handle:
        return json.loads(handle.read())


def completions(prompt: str, n: int, temperature: float, max_tokens: int = 4) -> list[dict]:
    body = post(
        SLOW,
        {
            "model": "qwen3",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": n,
            "logprobs": 1,
        },
    )
    out = []
    for choice in body.get("choices") or []:
        text = choice.get("text") or ""
        lp_nodes = (choice.get("logprobs") or {}).get("content") or []
        total = sum(float(x["logprob"]) for x in lp_nodes) if lp_nodes else 0.0
        out.append({"text": text, "slow_sum": total})
    return out


def fast_score(context: str, continuation: str) -> float:
    body = post(FAST_SCORE, {"context": context, "continuation": continuation})
    return float(body.get("logprob", float("-inf")))


def first_token_ok(text: str, truth: str) -> bool:
    return bool(text) and truth.startswith(text[:1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="n-best 重排序")
    ap.add_argument(
        "--cache", type=Path, default=Path("sweep_results/lm_fusion/alpha_v1_big.raw.json")
    )
    ap.add_argument("--n", type=int, default=4, help="llama-server 的 n_slots 是 4，超过会被 400")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    raw = json.loads(args.cache.read_text(encoding="utf-8"))
    if args.limit:
        raw = raw[: args.limit]

    dataset: list[dict] = []
    greedy_ok = fused_ok = oracle_ok = 0
    changed = 0
    n = 0
    for row in raw:
        context, truth = row["context"], row["truth"]
        try:
            greedy = completions(context, 1, 0.0)[0]
            cands = completions(context, args.n, 0.8)
        except Exception as exc:
            print(f"  跳过：{type(exc).__name__}", flush=True)
            continue
        if not cands:
            continue
        try:
            for c in cands:
                c["fast"] = fast_score(context, c["text"])
        except Exception as exc:
            print(f"  跳过打分：{type(exc).__name__}", flush=True)
            continue
        n += 1
        greedy_ok += int(first_token_ok(greedy["text"], truth))
        oracle_ok += int(any(first_token_ok(c["text"], truth) for c in cands))
        for c in cands:
            c["fused"] = c["slow_sum"] + args.alpha * c["fast"]
        slow_best = max(row["slow"], key=lambda kv: kv[1])[0] if row["slow"] else ""
        for c in cands:
            dataset.append(
                {
                    "slow_sum": round(c["slow_sum"], 4),
                    "fast_sum": round(c["fast"], 4),
                    "agree": int(bool(c["text"]) and c["text"][:1] == slow_best[:1]),
                    "label": int(first_token_ok(c["text"], truth)),
                }
            )
        best = max(cands, key=lambda c: c["fused"])
        fused_ok += int(first_token_ok(best["text"], truth))
        changed += int(best["text"][:1] != greedy["text"][:1])

    print(f"\n位置数 {n}")
    print(f"  千问 greedy 首个 token 命中   {greedy_ok:3d}  ({greedy_ok / max(n, 1):.3f})")
    print(f"  **融合重排序后命中**          {fused_ok:3d}  ({fused_ok / max(n, 1):.3f})")
    print(f"  候选集 oracle（上限）         {oracle_ok:3d}  ({oracle_ok / max(n, 1):.3f})")
    print(f"  重排序改变了首选的样本数      {changed}")
    print(f"\n  相对 greedy：{fused_ok - greedy_ok:+d} 个位置")
    dump = Path("sweep_results/lm_fusion/nbest_dataset.jsonl")
    dump.parent.mkdir(parents=True, exist_ok=True)
    with dump.open("w", encoding="utf-8", newline="\n") as handle:
        for item in dataset:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  数据集 {len(dataset)} 条 -> {dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
