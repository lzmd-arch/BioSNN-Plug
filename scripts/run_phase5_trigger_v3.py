"""阶段五触发判据 v3：零参数表面线索分数（判据见 sweep_results/phase5_trigger_v3_prereg.md）。

与 v2 的差别只有一处：分数从 `R5`（与最近 5 轮的余弦距离）换成一个**不用嵌入**的
表面线索计数。判据、α、划分、分位数取法、T-3 的基线全部照搬 v2 的 amend1，
由 `run_phase5_trigger_v2.evaluate_scores` 执行——同一套判决，只换输入。

这一版是**语料混杂检验**：如果一条零参数规则就过判据，那这 20 段测的是
「这轮有没有具体事实」，不是「该不该搜索」。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from run_phase5_trigger_v2 import (  # noqa: E402
    FPR_MAX,
    HIT_MIN,
    SIGNS_MIN,
    evaluate_scores,
    load_stream,
)

OUT = REPO / "sweep_results" / "phase5_trigger_v3"

#: 四类模式（预注册 §分数，跑之前写死，不增删）。
PATTERNS = (
    re.compile(r"\d+"),
    re.compile(
        r"[一二三四五六七八九十百千万零两]+"
        r"(?:月|日|号|点|分|楼|单元|层|岁|元|块|公里|小时|分钟)"
    ),
    re.compile(r"(?:周|星期)[一二三四五六日天]"),
    re.compile(r"“[^”]*”|\"[^\"]*\"|「[^」]*」"),
)


def score_surface(text: str) -> int:
    """`S(t)` = 四类模式在文本里的命中总次数。零参数、不做加权。"""
    return sum(len(pattern.findall(text)) for pattern in PATTERNS)


def score_conversations(stream: dict) -> list[list[int]]:
    return [
        [score_surface(turn["text"]) for turn in conversation["turns"]]
        for conversation in stream["conversations"]
    ]


def diagnostics(stream: dict, scores: list[list[int]], r5_aucs: list[float]) -> dict:
    """预注册 §诊断：四格分解、分数直方图、逐段 AUC（与 R5 并排）。"""
    from biosnn_mcp.roc import auc

    target_turns = set(stream["target_turns"])
    turns_per = stream["turns_per_conversation"]
    labels = [turn in target_turns for turn in range(turns_per)]
    cells = {"target_hit": 0, "target_miss": 0, "background_hit": 0, "background_miss": 0}
    histogram = {
        "target": {"0": 0, "1": 0, "2": 0, "3+": 0},
        "background": {"0": 0, "1": 0, "2": 0, "3+": 0},
    }
    per_conversation = []
    for index, row in enumerate(scores):
        for score, label in zip(row, labels, strict=True):
            side = "target" if label else "background"
            if label:
                cells["target_hit" if score > 0 else "target_miss"] += 1
            else:
                cells["background_hit" if score > 0 else "background_miss"] += 1
            bucket = "3+" if score >= 3 else str(score)
            histogram[side][bucket] += 1
        per_conversation.append(
            {
                "conversation": index,
                "auc_surface": auc(row, labels),
                "auc_r5": r5_aucs[index] if index < len(r5_aucs) else None,
            }
        )
    return {"cells": cells, "histogram": histogram, "per_conversation": per_conversation}


def load_r5_aucs(path: Path) -> list[float]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload.get("scores", []):
        if row.get("score") == "R5":
            return list(row.get("per_conversation_auc", []))
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="触发判据 v3 的表面线索分数")
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--expect-sha256", default="")
    parser.add_argument(
        "--r5-aucs",
        type=Path,
        default=REPO / "sweep_results" / "phase5_m3p" / "summary.json",
        help="已发布的 R5 逐段 AUC（只用于并排对照，不参与判据）",
    )
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)

    stream = load_stream(args.stream, args.expect_sha256)
    digest = hashlib.sha256(args.stream.read_bytes()).hexdigest()
    scores = score_conversations(stream)
    result = {
        "kind": "phase5 trigger v3: surface-cue score",
        "tag": args.tag or args.stream.stem,
        "stream_sha256": digest,
        "score": "surface-cue count (digits / cn-numerals+unit / weekday / quoted)",
        **evaluate_scores(stream, scores),
        "diagnostics": diagnostics(stream, scores, load_r5_aucs(args.r5_aucs)),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{result['tag']}.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )

    for name in ("A", "B"):
        block = result["directions"][name]
        print(
            f"[{result['tag']}] 方向 {name}: q={block['q']} 基线阈值={block['baseline_threshold']} "
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
    cells = result["diagnostics"]["cells"]
    print(f"[{result['tag']}] 四格: {cells}")
    print(f"[{result['tag']}] 直方图: {result['diagnostics']['histogram']}")
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
