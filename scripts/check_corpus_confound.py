"""语料混杂探针：在一份「轮级标签」的对话流上，量**零参数基线**能把标签分到多开。

起因是触发判据 v3：一条零参数表面线索规则就把三条判据全过了。这份脚本把它做成可复用的探针，
回答三个问题：

1. **位置**：目标轮在各段里是不是同一批下标？——若是，任何按位置猜的规则都满分，
   而这份语料就不再测「内容」了；
2. **长度**：只看这一轮多长；
3. **表面线索**：数字串 / 中文数字+单位 / 星期 / 引号；
4. **重复**：这一轮是不是与上一轮逐字相同（背景轮的常见形态）。

输出是每个基线的 AUC 与表面线索的四格分解。**它不判通过/不通过**——它只回答
「这份语料能不能被零参数规则解开」。用法::

    python scripts/check_corpus_confound.py --stream sweep_results/phase5_m3p/stream.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "biosnn-mcp" / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from biosnn_mcp.roc import auc  # noqa: E402
from run_phase5_trigger_v3 import score_surface  # noqa: E402


def load_stream(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def positional_report(stream: dict) -> dict:
    turns_per = stream["turns_per_conversation"]
    sets = [
        tuple(turn for turn in range(turns_per) if conversation["turns"][turn]["target"])
        for conversation in stream["conversations"]
    ]
    distinct = sorted(set(sets))
    return {
        "distinct_target_sets": len(distinct),
        "target_sets": [list(item) for item in distinct],
        "position_rule_accuracy": 1.0 if len(distinct) == 1 else None,
    }


def baseline_scores(stream: dict) -> dict[str, list[list[float]]]:
    scores: dict[str, list[list[float]]] = {"length": [], "surface": [], "not_repeat": []}
    for conversation in stream["conversations"]:
        texts = [turn["text"] for turn in conversation["turns"]]
        scores["length"].append([float(len(text)) for text in texts])
        scores["surface"].append([float(score_surface(text)) for text in texts])
        not_repeat = []
        for index, text in enumerate(texts):
            not_repeat.append(0.0 if index and text == texts[index - 1] else 1.0)
        scores["not_repeat"].append(not_repeat)
    return scores


def four_cells(scores: list[list[float]], labels: list[bool]) -> dict:
    cells = {"target_hit": 0, "target_miss": 0, "background_hit": 0, "background_miss": 0}
    for row in scores:
        for score, label in zip(row, labels, strict=True):
            key = ("target_" if label else "background_") + ("hit" if score > 0 else "miss")
            cells[key] += 1
    return cells


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="语料混杂探针")
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    stream = load_stream(args.stream)
    turns_per = stream["turns_per_conversation"]
    target_turns = set(stream["target_turns"])
    labels = [turn in target_turns for turn in range(turns_per)]

    positional = positional_report(stream)
    scores = baseline_scores(stream)
    rows = {}
    for name, per_conversation in scores.items():
        per_auc = [auc(row, labels) for row in per_conversation]
        pooled = [score for row in per_conversation for score in row]
        rows[name] = {
            "mean_auc": statistics.fmean(per_auc),
            "pooled_auc": auc(pooled, labels * len(per_conversation)),
            "per_conversation_auc": per_auc,
        }

    result = {
        "kind": "corpus confound probe",
        "stream": str(args.stream),
        "conversations": len(stream["conversations"]),
        "turns_per_conversation": turns_per,
        "target_turns": sorted(target_turns),
        "positional": positional,
        "baselines": rows,
        "surface_four_cells": four_cells(scores["surface"], labels),
    }

    print(f"段数 {result['conversations']} / 每段 {turns_per} 轮 / 目标轮 {sorted(target_turns)}")
    if positional["position_rule_accuracy"] == 1.0:
        print(
            f"[位置] 全部 {result['conversations']} 段的目标轮下标完全相同 "
            f"→ 一条「按下标猜」的规则准确率 {positional['position_rule_accuracy']:.0%}"
        )
    else:
        print(f"[位置] 目标下标有 {positional['distinct_target_sets']} 种，位置不构成完美预测")
    for name, row in rows.items():
        print(
            f"[基线] {name:10s} 池化 AUC {row['pooled_auc']:.3f} （逐段均值 {row['mean_auc']:.3f}）"
        )
    cells = result["surface_four_cells"]
    print(f"[四格] {cells}")

    if args.out is not None:
        args.out.write_text(
            json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
