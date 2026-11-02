"""阶段五触发判据 v2：conformal 触发评估（判据见 sweep_results/phase5_trigger_v2_prereg.md
与 phase5_trigger_v2_prereg_amend1.md）。

这一版**不换打分函数**（沿用 R5），测的是判据：把「打分好不好」（AUC）与「触发对不对」
（工作点的误触发率与命中率）分开报。校准集只用来定分位数，不选分数、不调任何超参。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "biosnn-mcp" / "src"))

from biosnn_mcp.embeddings import OllamaEmbedding  # noqa: E402
from biosnn_mcp.roc import auc, novelty_scores  # noqa: E402

OUT = REPO / "sweep_results" / "phase5_trigger_v2"

#: 判据（改这里必须改预注册或其 amend）。
ALPHA = 0.10
FPR_MAX = 0.10
HIT_MIN = 0.50
SIGNS_MIN = 12

#: 划分：偶数索引为校准、奇数为测试（amend1 第 2 节）。
CALIBRATION_PARITY = 0


def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open


def load_stream(path: Path, expected: str) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected and digest != expected:
        raise SystemExit(f"fixture 的 sha256 不符：{digest}（预期 {expected}）")
    return json.loads(path.read_text(encoding="utf-8"))


def conformal_quantile(scores: list[float], alpha: float) -> float:
    """split conformal 的分位数：升序第 ceil((n+1)(1-alpha)) 小（1-based）。"""
    if not scores:
        raise ValueError("校准池为空。")
    ordered = sorted(scores)
    n = len(ordered)
    index = math.ceil((n + 1) * (1.0 - alpha))
    if index > n:
        return float("inf")
    return ordered[index - 1]


def counts(scores: list[float], labels: list[bool], threshold: float) -> tuple[int, int, int]:
    tp = sum(1 for s, y in zip(scores, labels, strict=True) if y and s > threshold)
    fp = sum(1 for s, y in zip(scores, labels, strict=True) if not y and s > threshold)
    fn = sum(1 for s, y in zip(scores, labels, strict=True) if y and s <= threshold)
    return tp, fp, fn


def f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def pooled_f1(scores: list[float], labels: list[bool], threshold: float) -> float:
    return f1(*counts(scores, labels, threshold))


def best_threshold(scores: list[float], labels: list[bool]) -> float:
    """校准段上池化 F1 最大的固定阈值；并列取最小阈值。候选含两端。"""
    candidates = sorted(set(scores))
    candidates = [min(candidates) - 1.0, *candidates, max(candidates) + 1.0]
    best_value = -1.0
    best_at = candidates[0]
    for candidate in candidates:
        value = pooled_f1(scores, labels, candidate)
        if value > best_value + 1e-12:
            best_value = value
            best_at = candidate
    return best_at


def evaluate_scores(stream: dict, scores: list[list[float]]) -> dict:
    """判据与口径全在这里；分数由调用方给，所以 v3 能复用同一套判决。"""
    target_turns = set(stream["target_turns"])
    turns_per = stream["turns_per_conversation"]
    labels = [turn in target_turns for turn in range(turns_per)]

    indices = list(range(len(scores)))
    even = [index for index in indices if index % 2 == CALIBRATION_PARITY]
    odd = [index for index in indices if index % 2 != CALIBRATION_PARITY]

    directions = {}
    per_conversation = {index: {} for index in indices}
    for name, calib_indices, test_indices in (("A", even, odd), ("B", odd, even)):
        calib_scores = []
        calib_labels = []
        for index in calib_indices:
            calib_scores.extend(scores[index])
            calib_labels.extend(labels)
        negatives = [s for s, y in zip(calib_scores, calib_labels, strict=True) if not y]
        q = conformal_quantile(negatives, ALPHA)
        base = best_threshold(calib_scores, calib_labels)

        test_scores = []
        test_labels = []
        for index in test_indices:
            test_scores.extend(scores[index])
            test_labels.extend(labels)
        tp, fp, fn = counts(test_scores, test_labels, q)
        negatives_test = sum(1 for label in test_labels if not label)
        positives_test = sum(1 for label in test_labels if label)
        base_tp, base_fp, base_fn = counts(test_scores, test_labels, base)
        in_sample_fp = counts(calib_scores, calib_labels, q)[1]

        per_conversation_fpr = []
        for index in test_indices:
            conv_fp = counts(scores[index], labels, q)[1]
            per_conversation_fpr.append(conv_fp / sum(1 for label in labels if not label))

        for index in test_indices:
            conf_tp, conf_fp, conf_fn = counts(scores[index], labels, q)
            base_counts = counts(scores[index], labels, base)
            per_conversation[index] = {
                "direction": name,
                "f1_conformal": f1(conf_tp, conf_fp, conf_fn),
                "f1_baseline": f1(*base_counts),
                "tp": conf_tp,
                "fp": conf_fp,
                "fn": conf_fn,
            }

        directions[name] = {
            "calibration_indices": calib_indices,
            "test_indices": test_indices,
            "q": q,
            "baseline_threshold": base,
            "calibration_negatives": len(negatives),
            "in_sample_fpr": in_sample_fp / len(negatives),
            "test_negatives": negatives_test,
            "test_positives": positives_test,
            "false": fp,
            "fpr": fp / negatives_test,
            "hits": tp,
            "hit_rate": tp / positives_test,
            "misses": fn,
            "test_auc": auc(test_scores, test_labels),
            "baseline_test_f1": f1(base_tp, base_fp, base_fn),
            "conformal_test_f1": f1(tp, fp, fn),
            "per_conversation_fpr_median": statistics.median(per_conversation_fpr),
            "per_conversation_fpr_max": max(per_conversation_fpr),
            "test_conversations_above_alpha": sum(
                1 for value in per_conversation_fpr if value > ALPHA
            ),
        }

    primary = directions["A"]
    signs = sum(
        1
        for index in indices
        if per_conversation[index]["f1_conformal"] >= per_conversation[index]["f1_baseline"] - 1e-12
    )
    return {
        "stream_kind": stream.get("kind"),
        "conversations": len(scores),
        "turns_per_conversation": turns_per,
        "target_turns": sorted(target_turns),
        "alpha": ALPHA,
        "directions": directions,
        "per_conversation": [
            {"conversation": index, **per_conversation[index]} for index in indices
        ],
        "t1_fpr": primary["fpr"],
        "t1_pass": primary["fpr"] <= FPR_MAX,
        "t2_hit_rate": primary["hit_rate"],
        "t2_pass": primary["hit_rate"] >= HIT_MIN,
        "t3_signs": signs,
        "t3_total": len(indices),
        "t3_pass": signs >= SIGNS_MIN,
    }


def evaluate_stream(stream: dict, vectors: list[list[float]]) -> dict:
    """v2 的口径：分数 = 每段的 `R5`。"""
    return evaluate_scores(stream, [novelty_scores(conversation) for conversation in vectors])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="触发判据 v2 的 conformal 评估")
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--expect-sha256", default="")
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)

    stream = load_stream(args.stream, args.expect_sha256)
    digest = hashlib.sha256(args.stream.read_bytes()).hexdigest()
    embedder = OllamaEmbedding(model=args.model, host=args.host, timeout=240.0, opener=_opener())
    vectors = [
        embedder.embed([turn["text"] for turn in conversation["turns"]])
        for conversation in stream["conversations"]
    ]
    result = {
        "kind": "phase5 trigger v2: conformal gate",
        "tag": args.tag or args.stream.stem,
        "stream_sha256": digest,
        "model": args.model,
        **evaluate_stream(stream, vectors),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{result['tag']}.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )

    for name in ("A", "B"):
        block = result["directions"][name]
        print(
            f"[{result['tag']}] 方向 {name}: q={block['q']:.4f} "
            f"基线阈值={block['baseline_threshold']:.4f} "
            f"校准内误触发={block['in_sample_fpr']:.3f} "
            f"测试误触发={block['fpr']:.3f}（{block['false']}/{block['test_negatives']}） "
            f"命中={block['hit_rate']:.3f}（{block['hits']}/{block['test_positives']}） "
            f"测试 AUC={block['test_auc']:.3f}"
        )
    verdict = "过" if result["t1_pass"] else "不过"
    hit = "过" if result["t2_pass"] else "不过"
    sign_verdict = "过" if result["t3_pass"] else "不过"
    print(
        f"[{result['tag']}] T-1 {verdict} ({result['t1_fpr']:.3f} <= {FPR_MAX}) | "
        f"T-2 {hit} ({result['t2_hit_rate']:.3f} >= {HIT_MIN}) | "
        f"T-3 {sign_verdict} ({result['t3_signs']}/{result['t3_total']} >= {SIGNS_MIN})"
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
