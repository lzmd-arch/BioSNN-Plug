"""真实 50k 库的召回 profiling：建库、扫描、倒排候选。

用 `benchmarks/substrate` 的真实文本放大到 n 条，建一个 `MemoryStore`，量：
建库、`iter_items`、一次词法召回、倒排索引构建、候选 gather + Jaccard。
不修改 Axo 代码。
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


def build_index(items):
    bits = []
    rows = []
    for row, item in enumerate(items):
        if item.lex_sdr is None:
            continue
        bits.append(item.lex_sdr.idx)
        rows.append(np.full(item.lex_sdr.k, row, dtype=np.int64))
    flat_bits = np.concatenate(bits).astype(np.int64)
    flat_rows = np.concatenate(rows)
    order = np.argsort(flat_bits, kind="stable")
    sorted_bits = flat_bits[order]
    sorted_rows = flat_rows[order]
    counts = np.bincount(sorted_bits, minlength=4096)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return sorted_bits, sorted_rows, offsets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DEFAULT)
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--queries", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("sweep_results/axo_recall_50k/scan.json"))
    args = parser.parse_args(argv)

    axo = probe.Axo(args.axo_src)
    from memcore.recall import RecallConfig, recall
    from memcore.store import MemoryStore
    from memcore.types import MemoryItem, MemoryKind, Tier

    base_corpus = axo.load(args.corpus_dir)
    memories = axo.bench.scaled_memories(base_corpus.memories, args.n)
    started = time.perf_counter()
    contents = [memory.content for memory in memories]
    lex_codes = [axo.sdr.encode_lexical(text) for text in contents]
    encode_s = time.perf_counter() - started

    store = MemoryStore(":memory:")
    started = time.perf_counter()
    for index, (content, code) in enumerate(zip(contents, lex_codes, strict=True)):
        store.add(
            MemoryItem(
                memory_id=f"m{index:06d}",
                kind=MemoryKind.FACT,
                tier=Tier.RECENT,
                content=content,
                created_at=1_700_000_000.0 + index,
                lex_sdr=code,
            )
        )
    add_s = time.perf_counter() - started

    started = time.perf_counter()
    items = list(store.iter_items())
    iter_s = time.perf_counter() - started

    started = time.perf_counter()
    store.lexical_index()
    lexical_index_s = time.perf_counter() - started

    indices = np.unique(
        np.linspace(0, len(items) - 1, min(args.queries, len(items))).round().astype(int)
    )
    now = max(item.created_at for item in items) + 1.0
    recall_times = []
    scanned = []
    for index in indices:
        started = time.perf_counter()
        ledger = recall(
            store,
            query_text=items[int(index)].content,
            budget_tokens=10**9,
            config=RecallConfig(),
            now=now,
            settle=False,
        )
        recall_times.append(time.perf_counter() - started)
        scanned.append(ledger.scanned)

    started = time.perf_counter()
    _sorted_bits, sorted_rows, offsets = build_index(items)
    index_s = time.perf_counter() - started

    candidate_times = []
    candidate_sizes = []
    for index in indices:
        query = items[int(index)].lex_sdr
        started = time.perf_counter()
        gathered = [sorted_rows[offsets[bit] : offsets[bit + 1]] for bit in query.idx]
        candidates = (
            np.unique(np.concatenate(gathered)) if gathered else np.zeros(0, dtype=np.int64)
        )
        _scores = [items[int(row)].lex_sdr.jaccard(query) for row in candidates]
        candidate_times.append(time.perf_counter() - started)
        candidate_sizes.append(int(candidates.size))

    payload = {
        "n": len(items),
        "encode_s": encode_s,
        "add_s": add_s,
        "iter_s": iter_s,
        "recall_ms": float(np.mean(recall_times) * 1000),
        "recall_first_ms": float(recall_times[0] * 1000) if recall_times else None,
        "recall_warm_ms": float(np.mean(recall_times[1:]) * 1000)
        if len(recall_times) > 1
        else None,
        "recall_median_ms": float(np.median(recall_times) * 1000) if recall_times else None,
        "lexical_index_s": lexical_index_s,
        "scanned": int(np.mean(scanned)),
        "index_build_s": index_s,
        "candidate_ms": float(np.mean(candidate_times) * 1000),
        "candidates": float(np.mean(candidate_sizes)),
        "queries": len(indices),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(payload)
    print(f"[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
