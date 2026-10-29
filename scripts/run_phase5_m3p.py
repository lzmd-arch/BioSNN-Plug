"""阶段五 M3p：20 段语料 + 按难度分层（判据见 sweep_results/phase5_m3p_prereg.md）。

分层只用对照分数 `R5` 的段内 AUC 中位数（难/易各一半），不用任何一个待评估分数。
比较 `R5` / `Rmean` / `Rcentroid`，触发器固定参数只作参考。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "biosnn-mcp" / "src"))

from biosnn_mcp.embeddings import OllamaEmbedding  # noqa: E402
from biosnn_mcp.roc import auc, workpoint  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from run_phase5_m3n import (  # noqa: E402  # noqa: E402
    MS_PER_TURN,
    run_trigger,
    score_r5,
    score_rcentroid,
    score_rmean,
)

OUT = REPO / "sweep_results" / "phase5_m3p"

SCORES = {"R5": score_r5, "Rmean": score_rmean, "Rcentroid": score_rcentroid}

#: 判据（改这里必须改预注册文件）。
AUC_MIN = 0.80
PER_CONVERSATION_MIN = 0.70
CONVERSATIONS_MIN = 16
PAIRED_SIGNS_MIN = 15
HARD_POOLED_MIN = 0.75
HIT_WINDOW_MS = 200


def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open


def load_stream(path: Path, expected: str) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected and digest != expected:
        raise SystemExit(f"fixture 的 sha256 不符：{digest}（预期 {expected}）")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M3p 20 段语料的分层比较")
    parser.add_argument("--stream", type=Path, default=OUT / "stream.json")
    parser.add_argument("--expect-sha256", default="")
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    stream = load_stream(args.stream, args.expect_sha256)
    target_turns = list(stream["target_turns"])
    labels = [turn in target_turns for turn in range(stream["turns_per_conversation"])]
    embedder = OllamaEmbedding(model=args.model, host=args.host, timeout=240.0, opener=_opener())
    vectors = [
        embedder.embed([turn["text"] for turn in conversation["turns"]])
        for conversation in stream["conversations"]
    ]

    per_score: dict[str, list[float]] = {name: [] for name in SCORES}
    rule_scores = []
    trigger_rows = []
    for conversation_index, conversation in enumerate(vectors):
        for name, scorer in SCORES.items():
            scores = scorer(conversation)
            per_score[name].append(auc(scores, labels))
            if name == "R5":
                rule_scores = scores
        for seed in range(5):
            fired, z_scores = run_trigger(rule_scores, seed=seed)
            point = workpoint(
                fired, target_turns, ms_per_turn=MS_PER_TURN, hit_window_ms=HIT_WINDOW_MS
            )
            peaks = [
                max(z_scores[turn * MS_PER_TURN : (turn + 1) * MS_PER_TURN])
                for turn in range(len(z_scores) // MS_PER_TURN)
            ]
            trigger_rows.append(
                {
                    "conversation": conversation_index,
                    "seed": seed,
                    "hits": point["hits"],
                    "false": point["false"],
                    "trigger_auc": auc(peaks, labels),
                }
            )

    reference = per_score["R5"]
    median_reference = statistics.median(reference)
    hard = [index for index, value in enumerate(reference) if value < median_reference]
    easy = [index for index, value in enumerate(reference) if value >= median_reference]

    rows = []
    for name, values in per_score.items():
        pooled_all = statistics.fmean(values)
        hard_pooled = statistics.fmean([values[index] for index in hard])
        easy_pooled = statistics.fmean([values[index] for index in easy])
        signs = sum(1 for index, value in enumerate(values) if value > reference[index] + 1e-12)
        criteria = {
            "mean_auc": pooled_all,
            "conversations_at_or_above": sum(
                1 for value in values if value >= PER_CONVERSATION_MIN
            ),
            "criterion_1": pooled_all >= AUC_MIN
            and sum(1 for value in values if value >= PER_CONVERSATION_MIN) >= CONVERSATIONS_MIN,
            "paired_signs_vs_R5": signs,
            "criterion_2": name == "R5" or signs >= PAIRED_SIGNS_MIN,
            "hard_pooled": hard_pooled,
            "criterion_3": hard_pooled >= HARD_POOLED_MIN,
        }
        criteria["ready"] = bool(
            criteria["criterion_1"] and criteria["criterion_2"] and criteria["criterion_3"]
        )
        rows.append({"score": name, "per_conversation_auc": values, "criteria": criteria})
        print(
            f"{name:<10} mean {pooled_all:.3f}（≥0.70 的段 {criteria['conversations_at_or_above']}/"
            f"{len(values)}）难批 {hard_pooled:.3f} 易批 {easy_pooled:.3f} 对 R5 符号 {signs}",
            flush=True,
        )

    summary = {
        "kind": "phase5-m3p: 20 conversations with difficulty stratification",
        "stream_sha256": hashlib.sha256(args.stream.read_bytes()).hexdigest(),
        "conversations": len(vectors),
        "stratification": {
            "rule_score": "R5",
            "median": median_reference,
            "hard_indices": hard,
            "easy_indices": easy,
        },
        "criteria_thresholds": {
            "auc_min": AUC_MIN,
            "per_conversation_min": PER_CONVERSATION_MIN,
            "conversations_min": CONVERSATIONS_MIN,
            "paired_signs_min": PAIRED_SIGNS_MIN,
            "hard_pooled_min": HARD_POOLED_MIN,
        },
        "scores": rows,
        "trigger_reference": trigger_rows,
        "trigger_hits_median": statistics.median(row["hits"] for row in trigger_rows),
        "trigger_false_median": statistics.median(row["false"] for row in trigger_rows),
        "m3p_ready": any(row["criteria"]["ready"] for row in rows if row["score"] != "R5"),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"ready": summary["m3p_ready"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
