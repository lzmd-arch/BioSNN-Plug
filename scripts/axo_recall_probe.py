"""把新的关联码接进 Axo 的金标准召回链，扫描图通道权重与阈值。

预注册：[`sweep_results/axo_substrate_prereg.md`](../sweep_results/axo_substrate_prereg.md)
负责基质侧；本脚本是它的召回侧延伸，**只报告，不改 Axo 的任何默认值**。

用法::

    uv run python scripts/axo_recall_probe.py --arms idf-2.0,hybrid-24,hash
    uv run python scripts/axo_recall_probe.py --graph-weight 0,0.1,0.25 --min-score 0.10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AXO_SRC_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/src")
AXO_GOLD_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/benchmarks/gold")

#: 冻结基线的四项主指标（词法默认档）。
BASELINE = {
    "hit@1": 0.8961038961038961,
    "hit@8": 0.948051948051948,
    "mrr": 0.9095238095238095,
    "false_injection_rate": 0.8679245283018868,
    "auc": 0.9289143735081895,
}


class Axo:
    def __init__(self, src: Path) -> None:
        sys.path.insert(0, str(src))
        from axo.bench import goldset, harness, metrics
        from axo.bench import substrate as bench
        from memcore.graph import GraphConfig, GraphIndex
        from memcore.recall import RecallConfig

        self.goldset = goldset
        self.harness = harness
        self.metrics = metrics
        self.bench = bench
        self.GraphConfig = GraphConfig
        self.GraphIndex = GraphIndex
        self.RecallConfig = RecallConfig


def load_probe_module():
    """复用基质探针里的选位实现，避免两份选位规则漂移。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import axo_substrate_probe as probe

    return probe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--gold-dir", type=Path, default=AXO_GOLD_DEFAULT)
    parser.add_argument("--arms", default="idf-2.0,hybrid-24,hash")
    parser.add_argument("--graph-weight", default="0,0.1,0.25")
    parser.add_argument("--min-score", default="0.10")
    parser.add_argument("--decay", default="0.25")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--top-m", type=int, default=32)
    parser.add_argument("--min-weight", type=float, default=0.02)
    parser.add_argument(
        "--out", type=Path, default=Path("sweep_results/axo_recall_probe/scan.json")
    )
    args = parser.parse_args(argv)

    probe = load_probe_module()
    axo_probe = probe.Axo(args.axo_src)
    axo = Axo(args.axo_src)

    gold = axo.goldset.load_goldset(args.gold_dir)
    lex_codes = probe.lexical_codes(gold, axo_probe)
    df = probe.document_frequency(lex_codes, axo_probe.width)
    store = axo.harness.build_store(gold.memories)

    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    weights = [float(value) for value in args.graph_weight.split(",") if value.strip()]
    min_scores = [float(value) for value in args.min_score.split(",") if value.strip()]
    decays = [float(value) for value in args.decay.split(",") if value.strip()]

    rows = []
    for arm in arms:
        codes = probe.encode_arm(arm, lex_codes, df, len(gold.memories), args.k, axo_probe)
        coded = tuple(
            axo.bench.CodedMemory(
                memory_id=memory.memory_id,
                cluster=memory.kind,
                lex_sdr=lex,
                assoc_sdr=code,
            )
            for memory, lex, code in zip(gold.memories, lex_codes, codes, strict=True)
        )
        graph = axo.GraphIndex.build(
            coded,
            config=axo.GraphConfig(
                top_m=args.top_m, min_weight=args.min_weight, k=args.k, mode="hash"
            ),
        )
        for weight in weights:
            for min_score in min_scores:
                for decay in decays:
                    config = axo.RecallConfig(
                        graph_weight=weight, graph_decay=decay, min_score=min_score
                    )
                    outcomes = axo.harness.run(
                        gold, config=config, store=store, graph=graph if weight > 0 else None
                    )
                    summary = axo.metrics.summarize(outcomes)
                    row = {
                        "arm": arm,
                        "graph_weight": weight,
                        "min_score": min_score,
                        "decay": decay,
                        **{key: float(summary[key]) for key in BASELINE},
                    }
                    rows.append(row)
                    print(
                        f"{arm:<10} w={weight:<5} ms={min_score:<5} d={decay:<5} "
                        f"hit@1 {row['hit@1']:.4f}  hit@8 {row['hit@8']:.4f}  "
                        f"MRR {row['mrr']:.4f}  误注入 {row['false_injection_rate']:.4f}  "
                        f"AUC {row['auc']:.4f}"
                    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"baseline": BASELINE, "records": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\n[落盘] {args.out}")

    print("\n基线（词法默认档）：", {key: round(value, 4) for key, value in BASELINE.items()})
    survivors = [
        row
        for row in rows
        if row["hit@1"] >= BASELINE["hit@1"]
        and row["hit@8"] >= BASELINE["hit@8"]
        and row["mrr"] >= BASELINE["mrr"]
        and row["auc"] >= BASELINE["auc"]
        and row["false_injection_rate"] <= BASELINE["false_injection_rate"]
    ]
    print(f"五项全部不差的配置：{len(survivors)} 个")
    for row in survivors:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
