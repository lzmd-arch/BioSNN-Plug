"""50k 巩固：图构建、分簇、单趟 idf 回放的真实成本。

`tools/bench_consolidation.py` 每档同时跑 weight 与 idf 两趟；生产默认只跑 idf。
这里分开量：建图、partition、consolidate(code_mode="idf")，避免把 A/B 那趟算进
生产成本。语料仍是 substrate 放大到 50k。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import axo_substrate_probe as probe

AXO_SRC_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/src")
CORPUS_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/benchmarks/substrate")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DEFAULT)
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--out", type=Path, default=Path("sweep_results/axo_recall_50k/consolidation.json"))
    args = parser.parse_args(argv)

    axo = probe.Axo(args.axo_src)
    from memcore.consolidation import consolidate, partition
    from memcore.graph import GraphIndex
    from memcore.types import MemoryItem, MemoryKind, Tier

    base = axo.load(args.corpus_dir)
    memories = axo.bench.scaled_memories(base.memories, args.n)
    contents = [memory.content for memory in memories]
    started = time.perf_counter()
    codes = [axo.sdr.encode_lexical(text) for text in contents]
    encode_s = time.perf_counter() - started
    items = [
        MemoryItem(
            memory_id=memory.memory_id,
            kind=MemoryKind.FACT,
            tier=Tier.RECENT,
            content=content,
            created_at=1_700_000_000.0,
            lex_sdr=code,
        )
        for memory, content, code in zip(memories, contents, codes, strict=True)
    ]

    started = time.perf_counter()
    graph = GraphIndex.build(items)
    graph_s = time.perf_counter() - started

    started = time.perf_counter()
    clusters = partition(graph)
    partition_s = time.perf_counter() - started

    started = time.perf_counter()
    report = consolidate(graph, items, at=1_700_000_000.0 + args.n, code_mode="idf")
    consolidate_s = time.perf_counter() - started

    sizes = sorted((len(cluster) for cluster in report.clusters), reverse=True)
    payload = {
        "n": args.n,
        "encode_s": encode_s,
        "graph_s": graph_s,
        "nodes": len(graph),
        "edges": graph.edge_count(),
        "partition_s": partition_s,
        "clusters": len(clusters),
        "largest_cluster": sizes[0] if sizes else 0,
        "median_cluster": sizes[len(sizes) // 2] if sizes else 0,
        "consolidate_s": consolidate_s,
        "recognition_rate": report.recognition_rate,
        "replayed": report.replayed,
        "reinforced": len(report.reinforced),
        "missing_from_store": report.missing_from_store,
        "total_s": encode_s + graph_s + consolidate_s,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(payload)
    print(f"[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
