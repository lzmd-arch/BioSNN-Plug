"""50k 召回：稀有位候选 + 按 id 取条目的原型。

不改 Axo；只测：
- 全量词法 top-1 的成本；
- 按查询位的 df 从稀有位 gather 候选，候选大小 / top-1 一致率 / 耗时；
- 候选确定后 `store.get` 逐条取条目的成本（对比全量 iter_items）。
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DEFAULT)
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--queries", type=int, default=3)
    parser.add_argument("--rare-bits", default="8,16,32")
    parser.add_argument(
        "--out", type=Path, default=Path("sweep_results/axo_recall_rare_bit/scan.json")
    )
    args = parser.parse_args(argv)

    axo = probe.Axo(args.axo_src)
    from memcore.store import MemoryStore
    from memcore.types import MemoryItem, MemoryKind, Tier

    base_corpus = axo.load(args.corpus_dir)
    memories = axo.bench.scaled_memories(base_corpus.memories, args.n)
    contents = [memory.content for memory in memories]
    started = time.perf_counter()
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
    items = list(store.iter_items())
    started = time.perf_counter()
    flat_bits = np.concatenate([item.lex_sdr.idx for item in items]).astype(np.int64)
    flat_rows = np.concatenate(
        [np.full(item.lex_sdr.k, row, dtype=np.int64) for row, item in enumerate(items)]
    )
    order = np.argsort(flat_bits, kind="stable")
    sorted_bits = flat_bits[order]
    sorted_rows = flat_rows[order]
    counts = np.bincount(sorted_bits, minlength=4096)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    index_s = time.perf_counter() - started

    indices = np.unique(
        np.linspace(0, len(items) - 1, min(args.queries, len(items))).round().astype(int)
    )
    rows = []
    for index in indices:
        query = items[int(index)].lex_sdr
        started = time.perf_counter()
        full_scores = np.asarray([item.lex_sdr.jaccard(query) for item in items])
        full_order = np.argsort(-full_scores, kind="stable")
        full_ms = (time.perf_counter() - started) * 1000
        full_top = int(full_order[0])
        rare_order = query.idx[np.argsort(counts[query.idx], kind="stable")]
        for rare_count in [int(value) for value in args.rare_bits.split(",") if value.strip()]:
            started = time.perf_counter()
            chosen_bits = rare_order[:rare_count]
            candidates = np.unique(
                np.concatenate(
                    [sorted_rows[offsets[bit] : offsets[bit + 1]] for bit in chosen_bits]
                )
            )
            scores = np.asarray([items[int(row)].lex_sdr.jaccard(query) for row in candidates])
            candidate_top = int(candidates[int(np.argmax(scores))])
            candidate_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            for row in candidates[: min(len(candidates), 2000)]:
                store.get(items[int(row)].memory_id)
            get_ms = (time.perf_counter() - started) * 1000
            rows.append(
                {
                    "query": int(index),
                    "rare_bits": rare_count,
                    "candidates": int(candidates.size),
                    "top1_matches_full": bool(candidate_top == full_top),
                    "candidate_ms": candidate_ms,
                    "get_2000_ms": get_ms,
                    "full_ms": full_ms,
                    "full_top": full_top,
                }
            )
            print(
                f"q={index} rare={rare_count} cand={candidates.size} "
                f"top1_match={candidate_top == full_top} "
                f"cand {candidate_ms:.1f}ms get2000 {get_ms:.1f}ms full {full_ms:.1f}ms"
            )
    payload = {
        "n": len(items),
        "encode_s": encode_s,
        "add_s": add_s,
        "index_s": index_s,
        "records": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
