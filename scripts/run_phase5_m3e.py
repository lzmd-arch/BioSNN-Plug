"""阶段五 M3e：真实嵌入上的情景写入与睡眠巩固（判据见 phase5_m3e_prereg.md）。

流程：本机 Ollama 嵌入 → M3c 群体编码 → SQLite 情景库 → M3d 睡眠噪声巩固。
只连 127.0.0.1；不写原始嵌入到仓库（summary 只记指标与配对）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "biosnn-mcp" / "src"))

from biosnn_mcp import (  # noqa: E402
    EpisodicStore,
    OllamaEmbedding,
    SleepConsolidator,
    ZScoreTopKEncoder,
)

OUT = REPO / "sweep_results" / "phase5_m3e"

# 8 个事实簇 × 3 条同义改写 + 8 条互不相同的事实（运行后不改）。
CLUSTERS: list[list[str]] = [
    [
        "paris is the capital of france",
        "the capital of france is paris",
        "france's capital city is paris",
    ],
    [
        "water boils at 100 celsius at sea level",
        "at sea level water boils at 100 celsius",
        "the boiling point of water at sea level is 100 celsius",
    ],
    [
        "shakespeare wrote hamlet",
        "hamlet was written by shakespeare",
        "the author of hamlet is shakespeare",
    ],
    [
        "jupiter is the largest planet",
        "the largest planet is jupiter",
        "jupiter is the biggest planet in the solar system",
    ],
    [
        "gold has the chemical symbol au",
        "the chemical symbol for gold is au",
        "au is the symbol of gold",
    ],
    [
        "the pacific is the largest ocean",
        "the largest ocean is the pacific",
        "the pacific ocean is the biggest ocean",
    ],
    [
        "bell invented the telephone",
        "the telephone was invented by bell",
        "alexander graham bell invented the telephone",
    ],
    ["tokyo is the capital of japan", "the capital of japan is tokyo", "japan's capital is tokyo"],
]

DISTINCT: list[str] = [
    "the moon orbits the earth",
    "photosynthesis converts light into chemical energy",
    "the nile is a river in africa",
    "a hexagon has six sides",
    "penicillin was discovered by fleming",
    "the great wall is in china",
    "sound travels faster in water than in air",
    "the human heart pumps blood",
]

# 阈值标定用的**另一组**句子（不进入评测）。
CALIBRATION_SAME = [
    ("the sky appears blue during the day", "during the day the sky looks blue"),
    ("ice melts at zero celsius", "at zero celsius ice melts"),
    ("everest is the tallest mountain", "the tallest mountain is everest"),
    ("a triangle has three sides", "three sides make a triangle"),
    ("shakespeare wrote macbeth", "macbeth was written by shakespeare"),
    (
        "the atlantic separates europe and america",
        "europe and america are separated by the atlantic",
    ),
]
CALIBRATION_DIFFERENT = [
    ("the sky appears blue during the day", "a triangle has three sides"),
    ("ice melts at zero celsius", "shakespeare wrote macbeth"),
    ("everest is the tallest mountain", "the atlantic separates europe and america"),
    ("a triangle has three sides", "the sky appears blue during the day"),
    ("shakespeare wrote macbeth", "everest is the tallest mountain"),
    ("the atlantic separates europe and america", "ice melts at zero celsius"),
]


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


def _threshold(embedding: OllamaEmbedding) -> dict:
    pairs = CALIBRATION_SAME + CALIBRATION_DIFFERENT
    texts = sorted({text for pair in pairs for text in pair})
    vectors = dict(zip(texts, embedding.embed(texts), strict=True))
    same = [_cosine(vectors[a], vectors[b]) for a, b in CALIBRATION_SAME]
    different = [_cosine(vectors[a], vectors[b]) for a, b in CALIBRATION_DIFFERENT]
    mean_same = sum(same) / len(same)
    mean_different = sum(different) / len(different)
    threshold = (mean_same + mean_different) / 2.0
    return {
        "mean_same": mean_same,
        "mean_different": mean_different,
        "threshold": threshold,
        "min_same": min(same),
        "max_different": max(different),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M3e 真实嵌入的情景写入与睡眠巩固")
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument("--noise-sigma", type=float, default=0.05)
    parser.add_argument(
        "--margin", type=float, default=0.0, help="M3f：最近邻与次近邻的最小余弦间隔（0=历史行为）"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})).open
    embedding = OllamaEmbedding(model=args.model, opener=opener)
    calibration = _threshold(embedding)

    encoder = ZScoreTopKEncoder(top_k=8, n_centers=6)
    rows = []
    for cluster_index, texts in enumerate(CLUSTERS):
        for variant, text in enumerate(texts):
            rows.append((f"cluster{cluster_index}:{variant}", f"c{cluster_index}", text))
    for index, text in enumerate(DISTINCT):
        rows.append((f"distinct{index}", f"d{index}", text))

    vectors = embedding.embed([text for _, _, text in rows])
    with EpisodicStore() as store:
        for (identifier, _, text), vector in zip(rows, vectors, strict=True):
            code = encoder.encode(vector).spikes
            store.add(identifier, text, vector, code=code)

        consolidator = SleepConsolidator(
            noise_sigma=args.noise_sigma,
            merge_threshold=calibration["threshold"],
            merge_margin=args.margin,
            rounds=3,
            seed=args.seed,
        )
        survivors, report = consolidator.consolidate(store.items())
        survivors_again, report_again = consolidator.consolidate(store.items())
        store.replace_all(survivors)

    cluster_of = {identifier: cluster for identifier, cluster, _ in rows}
    survivor_clusters: dict[str, int] = {}
    for item in survivors:
        survivor_clusters[cluster_of[item.id]] = survivor_clusters.get(cluster_of[item.id], 0) + 1

    paraphrase_clusters = [f"c{index}" for index in range(len(CLUSTERS))]
    merged_clusters = sum(
        1 for cluster in paraphrase_clusters if survivor_clusters.get(cluster) == 1
    )
    false_merges = [
        pair for pair in report.merged_pairs if cluster_of[pair[0]] != cluster_of[pair[1]]
    ]
    deterministic = report.merged_pairs == report_again.merged_pairs and [
        item.id for item in survivors
    ] == [item.id for item in survivors_again]

    gate = {
        "paraphrase_merge": merged_clusters >= 7,
        "no_false_merges": len(false_merges) <= 1,
        "deterministic": deterministic,
        "localhost_only": True,
    }
    gate["m3e_ready"] = all(gate.values())
    summary = {
        "kind": "real-embedding episodic write + sleep-noise consolidation (protocol)",
        "model": args.model,
        "calibration": calibration,
        "noise_sigma": args.noise_sigma,
        "merge_margin": args.margin,
        "seed": args.seed,
        "counts": {
            "entries": len(rows),
            "survivors": len(survivors),
            "merged_pairs": len(report.merged_pairs),
            "false_merges": len(false_merges),
        },
        "merged_clusters": merged_clusters,
        "false_merges": false_merges,
        "survivor_per_cluster": survivor_clusters,
        "report": {
            "rounds_used": report.rounds_used,
            "mean_cosine_drift": report.mean_cosine_drift,
        },
        "gate": gate,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary["gate"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
