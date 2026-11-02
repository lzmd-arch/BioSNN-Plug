"""触发判据 v2 的补充诊断（**在看到 T-1 的 A/B 分歧之后**写的，只用于解释，不参与判据）。

回答两个问题：

1. T-4 的合成对照流为什么没按设计工作（目标轮与背景轮的 R5 分开度）；
2. 真实流上 T-1 的 0.153 / 0.067 是「分布漂移」还是「单个 10/10 划分的抽样噪声」——
   把划分随机重做 `--splits` 次，看误触发率与命中率的分布。

判据、α、划分规则一个都不动；这份脚本不改 v2 的判决。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "biosnn-mcp" / "src"))

from biosnn_mcp.embeddings import OllamaEmbedding  # noqa: E402
from biosnn_mcp.roc import novelty_scores  # noqa: E402

ALPHA = 0.10
SEED = 20261102


def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open


def conformal_quantile(scores: list[float], alpha: float) -> float:
    ordered = sorted(scores)
    n = len(ordered)
    index = math.ceil((n + 1) * (1.0 - alpha))
    return float("inf") if index > n else ordered[index - 1]


def split_metrics(scores, labels, calib, test):
    calib_scores = []
    calib_labels = []
    for index in calib:
        calib_scores.extend(scores[index])
        calib_labels.extend(labels)
    negatives = [s for s, y in zip(calib_scores, calib_labels, strict=True) if not y]
    q = conformal_quantile(negatives, ALPHA)
    false = hits = negatives_test = positives_test = 0
    for index in test:
        for score, label in zip(scores[index], labels, strict=True):
            fired = score > q
            if label:
                positives_test += 1
                hits += fired
            else:
                negatives_test += 1
                false += fired
    return false / negatives_test, hits / positives_test


def describe(values: list[float]) -> dict:
    ordered = sorted(values)
    low = ordered[int(0.05 * (len(ordered) - 1))]
    high = ordered[int(0.95 * (len(ordered) - 1))]
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values),
        "median": statistics.median(values),
        "p05": low,
        "p95": high,
        "min": ordered[0],
        "max": ordered[-1],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="触发判据 v2 的补充诊断")
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--expect-sha256", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--splits", type=int, default=500)
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--out", type=Path, default=REPO / "sweep_results" / "phase5_trigger_v2")
    args = parser.parse_args(argv)

    digest = hashlib.sha256(args.stream.read_bytes()).hexdigest()
    if args.expect_sha256 and digest != args.expect_sha256:
        raise SystemExit(f"fixture 的 sha256 不符：{digest}")
    stream = json.loads(args.stream.read_text(encoding="utf-8"))
    target_turns = set(stream["target_turns"])
    labels = [turn in target_turns for turn in range(stream["turns_per_conversation"])]
    embedder = OllamaEmbedding(model=args.model, host=args.host, timeout=240.0, opener=_opener())
    scores = [
        novelty_scores(embedder.embed([turn["text"] for turn in conversation["turns"]]))
        for conversation in stream["conversations"]
    ]

    positives = [score for row in scores for score, y in zip(row, labels, strict=True) if y]
    negatives = [score for row in scores for score, y in zip(row, labels, strict=True) if not y]
    separation = {
        "target_mean": statistics.fmean(positives),
        "target_median": statistics.median(positives),
        "background_mean": statistics.fmean(negatives),
        "background_median": statistics.median(negatives),
    }

    rng = random.Random(SEED)
    indices = list(range(len(scores)))
    fprs = []
    hits = []
    for _ in range(args.splits):
        shuffled = indices[:]
        rng.shuffle(shuffled)
        calib, test = shuffled[: len(indices) // 2], shuffled[len(indices) // 2 :]
        fpr, hit = split_metrics(scores, labels, calib, test)
        fprs.append(fpr)
        hits.append(hit)

    result = {
        "kind": "phase5 trigger v2: post-hoc diagnostic (not a criterion)",
        "tag": args.tag or args.stream.stem,
        "stream_sha256": digest,
        "alpha": ALPHA,
        "splits": args.splits,
        "split_seed": SEED,
        "separation": separation,
        "fpr": {
            **describe(fprs),
            "share_at_or_below_alpha": sum(f <= ALPHA for f in fprs) / len(fprs),
        },
        "hit_rate": {
            **describe(hits),
            "share_at_or_above_half": sum(h >= 0.5 for h in hits) / len(hits),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{result['tag']}_diagnostic.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )

    print(
        f"[{result['tag']}] 目标轮 R5 均值 {separation['target_mean']:.4f} / "
        f"背景轮 {separation['background_mean']:.4f}（差 {separation['target_mean'] - separation['background_mean']:+.4f}）"
    )
    print(
        f"[{result['tag']}] 随机划分 {args.splits} 次："
        f"误触发率 均值 {result['fpr']['mean']:.3f} 中位 {result['fpr']['median']:.3f} "
        f"[{result['fpr']['p05']:.3f}, {result['fpr']['p95']:.3f}] "
        f"≤α 的划分占 {result['fpr']['share_at_or_below_alpha']:.2f}"
    )
    print(
        f"[{result['tag']}]                  "
        f"命中率   均值 {result['hit_rate']['mean']:.3f} 中位 {result['hit_rate']['median']:.3f} "
        f"[{result['hit_rate']['p05']:.3f}, {result['hit_rate']['p95']:.3f}] "
        f"≥0.50 的划分占 {result['hit_rate']['share_at_or_above_half']:.2f}"
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
