"""50k 速率路径：图邻居扩展候选。

预注册：[`sweep_results/axo_rate_graph_expand_prereg.md`](../sweep_results/axo_rate_graph_expand_prereg.md)。

一阶分数取种子 -> 关联图邻居扩展 -> 子集二阶分数。图构建时间单独报。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import axo_spiking_prototype as base
import axo_spiking_sparse_prototype as sparse_mod
import axo_substrate_probe as probe

AXO_SRC_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/src")
CORPUS_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/benchmarks/substrate")


class Coded:
    def __init__(self, memory_id: str, code) -> None:
        self.memory_id = memory_id
        self.assoc_sdr = code
        self.lex_sdr = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DEFAULT)
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--probes", type=int, default=20)
    parser.add_argument(
        "--out", type=Path, default=Path("sweep_results/axo_rate_graph_expand/scan.json")
    )
    args = parser.parse_args(argv)

    axo = probe.Axo(args.axo_src)
    base_corpus = axo.load(args.corpus_dir)
    memories = axo.bench.scaled_memories(base_corpus.memories, args.n)
    corpus = axo.bench.SubstrateCorpus(
        memories=tuple(memories), queries=base_corpus.queries, corpus_dir=base_corpus.corpus_dir
    )
    lex = probe.lexical_codes(corpus, axo)
    df = probe.document_frequency(lex, axo.width)
    codes = probe.encode_arm("idf-2.0", lex, df, args.n, 64, axo)
    sparse = sparse_mod.sparse_from_codes(codes, axo.width)

    from memcore.graph import GraphConfig, GraphIndex

    coded = [Coded(str(index), code) for index, code in enumerate(codes)]
    started = time.perf_counter()
    graph = GraphIndex.build(coded, config=GraphConfig(top_m=8, min_weight=0.10, k=64, mode="hash"))
    build_s = time.perf_counter() - started

    indices = np.unique(np.linspace(0, args.n - 1, min(args.probes, args.n)).round().astype(int))
    probes = [
        base.dense_probe(axo.sdr.drop_bits(codes[int(index)], 0.5), axo.width) for index in indices
    ]
    started = time.perf_counter()
    full_orders = [sparse_mod.rate_static_sparse(sparse, item) for item in probes]
    full_ms = (time.perf_counter() - started) / len(probes) * 1000

    rows = []
    for m0, m1 in ((2000, 5000), (2000, 10000), (5000, 10000), (5000, 15000)):
        correct = 0
        times = []
        counts = []
        for item, full in zip(probes, full_orders, strict=True):
            started = time.perf_counter()
            first = sparse.matvec(item)
            seeds = np.argsort(-first, kind="stable")[:m0]
            candidates = {int(index) for index in seeds}
            for seed in seeds:
                node_id = str(int(seed))
                for neighbor_id, _weight in graph.neighbors(node_id):
                    candidates.add(int(neighbor_id))
            ordered = sorted(candidates, key=lambda index: -float(first[index]))[:m1]
            subset = sparse.subset(np.asarray(ordered, dtype=np.int64))
            activation = subset.transpose_matvec(subset.matvec(item))
            order = np.argsort(-subset.matvec(activation), kind="stable")
            times.append(time.perf_counter() - started)
            counts.append(len(ordered))
            correct += int(int(ordered[int(order[0])]) == int(full[0]))
        row = {
            "m0": m0,
            "m1": m1,
            "agreement": correct / len(probes),
            "mean_candidates": float(np.mean(counts)),
            "mean_ms": float(np.mean(times) * 1000),
            "p95_ms": float(np.percentile(times, 95) * 1000),
        }
        rows.append(row)
        print(
            f"M0={m0} M1={m1}: agree {row['agreement']:.4f} "
            f"cand {row['mean_candidates']:.0f} mean {row['mean_ms']:.2f}ms "
            f"p95 {row['p95_ms']:.2f}ms"
        )
    payload = {
        "n": args.n,
        "graph_build_s": build_s,
        "full_ms": full_ms,
        "probes": len(probes),
        "records": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"graph build {build_s:.1f}s full {full_ms:.2f}ms; [落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
