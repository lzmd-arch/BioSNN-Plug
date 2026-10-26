"""关联图建图加速原型：稀疏 posting 的精确 Jaccard vs 现有稠密分块矩阵。

关联码每条只有 64 个活跃位，但现有 ``GraphIndex.build`` 用 4096 宽的稠密指示矩阵
做分块乘法。这里用同样的关联码、同样的 top-m/阈值，改成按位 posting 取交集，
比较**边是否逐条相同**与建图耗时。

只建图，不跑召回；语料是 substrate 按 ``scaled_memories`` 放大。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import axo_substrate_probe as probe

AXO_SRC_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/src")
CORPUS_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/benchmarks/substrate")


def sparse_build(patterns, width, config):
    """按位 posting 的精确 Jaccard top-m；语义照 ``GraphIndex._select_top_m``。"""
    from memcore.graph import GraphIndex

    size = len(patterns)
    cardinality = np.asarray([sdr.k for sdr in patterns], dtype=np.int32)
    postings = [[] for _ in range(width)]
    for row, sdr in enumerate(patterns):
        for bit in sdr.idx:
            postings[int(bit)].append(row)
    postings = [np.asarray(rows, dtype=np.int32) for rows in postings]

    nbr_idx = np.full((size, config.top_m), -1, dtype=np.int32)
    nbr_w = np.zeros((size, config.top_m), dtype=np.float32)
    for row, sdr in enumerate(patterns):
        blocks = [postings[int(bit)] for bit in sdr.idx if postings[int(bit)].size]
        if not blocks:
            continue
        counts = np.bincount(np.concatenate(blocks), minlength=size).astype(np.int32)
        touched = np.flatnonzero(counts)
        inter = counts[touched]
        union = cardinality[row] + cardinality[touched] - inter
        similarity = (inter / np.maximum(union, 1)).astype(np.float32)
        valid = (touched != row) & (similarity >= config.min_weight) & (similarity > 0.0)
        touched = touched[valid]
        similarity = similarity[valid]
        if touched.size == 0:
            continue
        order = np.lexsort((touched, -similarity))[: config.top_m]
        picked = touched[order]
        nbr_idx[row, : picked.size] = picked
        nbr_w[row, : picked.size] = similarity[order]

    # 用同一份 node_ids/指纹，便于逐条比较；这里只关心边。
    return GraphIndex(
        node_ids=tuple(f"n{index}" for index in range(size)),
        nbr_idx=nbr_idx,
        nbr_w=nbr_w,
        source_digest="prototype",
        built_at=0.0,
        top_m=config.top_m,
        min_weight=config.min_weight,
        assoc_k=config.k,
        assoc_mode=config.mode,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DEFAULT)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument(
        "--out", type=Path, default=Path("sweep_results/axo_graph_sparse/scan.json")
    )
    args = parser.parse_args(argv)

    axo = probe.Axo(args.axo_src)
    from memcore.graph import GraphConfig, GraphIndex, _extract
    from memcore.types import MemoryItem, MemoryKind, Tier

    base = axo.load(args.corpus_dir)
    memories = axo.bench.scaled_memories(base.memories, args.n)
    items = []
    for memory in memories:
        items.append(
            MemoryItem(
                memory_id=memory.memory_id,
                kind=MemoryKind.FACT,
                tier=Tier.RECENT,
                content=memory.content,
                created_at=1_700_000_000.0,
                lex_sdr=axo.sdr.encode_lexical(memory.content),
            )
        )
    config = GraphConfig()

    started = time.perf_counter()
    full = GraphIndex.build(items, config=config)
    full_s = time.perf_counter() - started

    started = time.perf_counter()
    node_ids, patterns, width = _extract(items, k=config.k, mode=config.mode)
    extract_s = time.perf_counter() - started

    started = time.perf_counter()
    sparse = sparse_build(patterns, width, config)
    sparse_s = time.perf_counter() - started

    edge_same = int(np.array_equal(full.nbr_idx, sparse.nbr_idx))
    mismatch = int(np.count_nonzero(full.nbr_idx != sparse.nbr_idx))
    weight_max_diff = float(np.max(np.abs(full.nbr_w - sparse.nbr_w)))
    payload = {
        "n": args.n,
        "width": width,
        "full_s": full_s,
        "extract_s": extract_s,
        "sparse_s": sparse_s,
        "sparse_total_s": extract_s + sparse_s,
        "speedup": full_s / (extract_s + sparse_s) if extract_s + sparse_s else None,
        "edges": full.edge_count(),
        "nbr_idx_equal": bool(edge_same),
        "nbr_idx_mismatch": mismatch,
        "weight_max_diff": weight_max_diff,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(payload)
    print(f"[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
