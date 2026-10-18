"""速率路径的倒排候选子集验证：真实文本放大语料上的正确性与延迟。

预注册：[`sweep_results/axo_rate_candidate_validation_prereg.md`](../sweep_results/axo_rate_candidate_validation_prereg.md)。

真实库不参与；语料是 `benchmarks/substrate` 的真实文本按 `scaled_memories` 放大。
输出每个 (n, M) 的一致率、延迟、候选数；不建持久索引。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import axo_spiking_prototype as spiking_base
import axo_spiking_sparse_prototype as sparse_mod
import axo_substrate_probe as probe

AXO_SRC_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/src")
CORPUS_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/benchmarks/substrate")


def build_inverted(codes: sparse_mod.SparseCodes):
    flat = codes.idx.ravel().astype(np.int64)
    rows = np.repeat(np.arange(codes.n, dtype=np.int64), codes.idx.shape[1])
    order = np.argsort(flat, kind="stable")
    sorted_bits = flat[order]
    sorted_rows = rows[order]
    counts = np.bincount(flat, minlength=codes.width)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return sorted_bits, sorted_rows, codes.weights.ravel()[order], offsets


def candidates(
    probe_vector, sorted_bits, sorted_rows, sorted_weights, offsets, *, limit: int
) -> np.ndarray:
    """按**真实稀疏点积**排候选，而不是按共享位数。

    共享位数会漏掉权重大但只共享少数位的那条；点积与全量扫描的排序同源，候选集
    里只要包含目标，截断就不会把它挤掉。
    """
    bits = np.flatnonzero(probe_vector)
    rows = [sorted_rows[offsets[bit] : offsets[bit + 1]] for bit in bits]
    weights = [sorted_weights[offsets[bit] : offsets[bit + 1]] * probe_vector[bit] for bit in bits]
    if not rows:
        return np.zeros(0, dtype=np.int64)
    merged = np.concatenate(rows)
    scores = np.bincount(
        merged, weights=np.concatenate(weights), minlength=int(sorted_rows.max()) + 1
    )
    return np.argsort(-scores, kind="stable")[:limit]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DEFAULT)
    parser.add_argument("--scales", default="200,500,1000,2000,10000")
    parser.add_argument("--limits", default="100,200,500,1000,2000,5000")
    parser.add_argument("--probes", type=int, default=100)
    parser.add_argument(
        "--out", type=Path, default=Path("sweep_results/axo_rate_candidate/scan.json")
    )
    args = parser.parse_args(argv)

    axo = probe.Axo(args.axo_src)
    base = axo.load(args.corpus_dir)
    rows = []
    for n in [int(value) for value in args.scales.split(",") if value.strip()]:
        memories = axo.bench.scaled_memories(base.memories, n)
        corpus = axo.bench.SubstrateCorpus(
            memories=tuple(memories), queries=base.queries, corpus_dir=base.corpus_dir
        )
        lex = probe.lexical_codes(corpus, axo)
        df = probe.document_frequency(lex, axo.width)
        codes = probe.encode_arm("idf-2.0", lex, df, n, 64, axo)
        sparse = sparse_mod.sparse_from_codes(codes, axo.width)
        started = time.perf_counter()
        sorted_bits, sorted_rows, sorted_weights, offsets = build_inverted(sparse)
        build_s = time.perf_counter() - started

        count = min(args.probes, n)
        indices = np.unique(np.linspace(0, n - 1, count).round().astype(np.int64))
        probes = [
            spiking_base.dense_probe(axo.sdr.drop_bits(codes[int(index)], 0.5), axo.width)
            for index in indices
        ]
        started = time.perf_counter()
        full_orders = [sparse_mod.rate_static_sparse(sparse, item) for item in probes]
        full_ms = (time.perf_counter() - started) / len(probes) * 1000
        for limit in [int(value) for value in args.limits.split(",") if value.strip()]:
            correct = top5 = 0
            times = []
            counts = []
            for item, full in zip(probes, full_orders, strict=True):
                started = time.perf_counter()
                chosen = candidates(
                    item, sorted_bits, sorted_rows, sorted_weights, offsets, limit=limit
                )
                if chosen.size == 0:
                    times.append(time.perf_counter() - started)
                    counts.append(0)
                    continue
                subset = sparse.subset(chosen)
                activation = subset.transpose_matvec(subset.matvec(item))
                order = np.argsort(-subset.matvec(activation), kind="stable")
                times.append(time.perf_counter() - started)
                counts.append(int(chosen.size))
                best = int(chosen[int(order[0])])
                correct += int(best == int(full[0]))
                top5 += int(int(full[0]) in [int(value) for value in chosen[order[:5]]])
            row = {
                "n": n,
                "limit": limit,
                "probes": len(probes),
                "top1_agreement": correct / len(probes),
                "top5_agreement": top5 / len(probes),
                "query_ms": float(np.mean(times) * 1000),
                "candidates_mean": float(np.mean(counts)),
                "build_s": build_s,
                "full_ms": full_ms,
            }
            rows.append(row)
            print(
                f"n={n:<6} M={limit:<5} agree {row['top1_agreement']:.4f} "
                f"top5 {row['top5_agreement']:.4f} "
                f"q {row['query_ms']:.2f}ms cand {row['candidates_mean']:.0f} "
                f"(full {full_ms:.2f}ms build {build_s:.2f}s)"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"records": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
