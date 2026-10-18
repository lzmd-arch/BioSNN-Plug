"""50k 速率路径：固定候选 vs 自适应候选。

预注册：[`sweep_results/axo_rate_adaptive_prereg.md`](../sweep_results/axo_rate_adaptive_prereg.md)。

只在真实文本放大语料上做推理加速；不涉及脉冲、学习或 Axo 代码。
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


def candidate_scores(
    sparse: sparse_mod.SparseCodes, probe_vector: np.ndarray, limit: int
) -> tuple[np.ndarray, np.ndarray, float]:
    first = sparse.matvec(probe_vector)
    chosen = np.argsort(-first, kind="stable")[:limit]
    subset = sparse.subset(chosen)
    activation = subset.transpose_matvec(subset.matvec(probe_vector))
    scores = subset.matvec(activation)
    order = np.argsort(-scores, kind="stable")
    top = chosen[order]
    return top, scores[order], float(first.max())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DEFAULT)
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--probes", type=int, default=50)
    parser.add_argument("--m0", type=int, default=10000)
    parser.add_argument("--m1", type=int, default=15000)
    parser.add_argument("--taus", default="0.05,0.10,0.20")
    parser.add_argument(
        "--out", type=Path, default=Path("sweep_results/axo_rate_adaptive/scan.json")
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
    indices = np.unique(np.linspace(0, args.n - 1, min(args.probes, args.n)).round().astype(int))
    probes = [
        base.dense_probe(axo.sdr.drop_bits(codes[int(index)], 0.5), axo.width) for index in indices
    ]

    started = time.perf_counter()
    full_orders = [sparse_mod.rate_static_sparse(sparse, item) for item in probes]
    full_ms = (time.perf_counter() - started) / len(probes) * 1000
    rows = []
    for limit in (args.m0, args.m1):
        correct = 0
        times = []
        for item, full in zip(probes, full_orders, strict=True):
            started = time.perf_counter()
            top, _scores, _peak = candidate_scores(sparse, item, limit)
            times.append(time.perf_counter() - started)
            correct += int(int(top[0]) == int(full[0]))
        rows.append(
            {
                "arm": "fixed",
                "m": limit,
                "agreement": correct / len(probes),
                "mean_ms": float(np.mean(times) * 1000),
                "p95_ms": float(np.percentile(times, 95) * 1000),
            }
        )
        print(
            f"fixed M={limit}: agree {correct / len(probes):.4f} "
            f"mean {rows[-1]['mean_ms']:.2f}ms p95 {rows[-1]['p95_ms']:.2f}ms"
        )

    for tau in [float(value) for value in args.taus.split(",") if value.strip()]:
        correct = 0
        expanded = 0
        times = []
        chosen_counts = []
        for item, full in zip(probes, full_orders, strict=True):
            started = time.perf_counter()
            top, scores, _peak = candidate_scores(sparse, item, args.m0)
            margin = (scores[0] - scores[1]) / max(abs(scores[0]), 1e-9) if scores.size > 1 else 1.0
            used = args.m0
            if margin < tau:
                expanded += 1
                top, scores, _peak = candidate_scores(sparse, item, args.m1)
                used = args.m1
            times.append(time.perf_counter() - started)
            chosen_counts.append(used)
            correct += int(int(top[0]) == int(full[0]))
        row = {
            "arm": "adaptive",
            "tau": tau,
            "m0": args.m0,
            "m1": args.m1,
            "agreement": correct / len(probes),
            "expanded_fraction": expanded / len(probes),
            "mean_m": float(np.mean(chosen_counts)),
            "mean_ms": float(np.mean(times) * 1000),
            "p95_ms": float(np.percentile(times, 95) * 1000),
        }
        rows.append(row)
        print(
            f"adaptive tau={tau:.2f}: agree {row['agreement']:.4f} "
            f"expand {row['expanded_fraction']:.2f} meanM {row['mean_m']:.0f} "
            f"mean {row['mean_ms']:.2f}ms p95 {row['p95_ms']:.2f}ms"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"n": args.n, "full_ms": full_ms, "records": rows},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"full {full_ms:.2f}ms; [落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
