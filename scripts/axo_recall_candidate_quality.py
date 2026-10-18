"""50k 词法候选质量：稀有位 gather vs 按精确 Jaccard 排序。

只读研究脚本：建一个 50k 的 `MemoryStore`（真实文本放大），用 substrate 的 40 条
查询与 expect 集，量候选覆盖率、top-1 一致率、hit@1/@8/MRR 与耗时。不改 Axo。
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


def exact_counts(index, query):
    rows = []
    for bit in query.idx:
        bit = int(bit)
        if 0 <= bit + 1 < index.offsets.size:
            rows.append(index.rows[index.offsets[bit] : index.offsets[bit + 1]])
    if not rows:
        return np.zeros(len(index.ids), dtype=np.int64)
    return np.bincount(np.concatenate(rows), minlength=len(index.ids))


def ranked_exact(index, query, lengths, limit):
    counts = exact_counts(index, query)
    touched = np.flatnonzero(counts)
    if touched.size == 0:
        return touched
    inter = counts[touched]
    union = query.k + lengths[touched] - inter
    score = inter / np.maximum(union, 1)
    order = np.argsort(-score, kind="stable")[:limit]
    return touched[order]


def metrics(ranked_ids, expect):
    expect = set(expect)
    if not expect:
        return {"hit1": None, "hit8": None, "mrr": None}
    first = None
    for position, memory_id in enumerate(ranked_ids, start=1):
        if memory_id in expect:
            first = position
            break
    return {
        "hit1": 1.0 if ranked_ids and ranked_ids[0] in expect else 0.0,
        "hit8": 1.0 if first is not None and first <= 8 else 0.0,
        "mrr": 0.0 if first is None else 1.0 / first,
    }


def mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DEFAULT)
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--limits", default="1000,2000,5000,10000")
    parser.add_argument("--out", type=Path, default=Path("sweep_results/axo_recall_50k/candidate_quality.json"))
    args = parser.parse_args(argv)

    axo = probe.Axo(args.axo_src)
    from memcore.store import MemoryStore
    from memcore.types import MemoryItem, MemoryKind, Tier

    base = axo.load(args.corpus_dir)
    memories = axo.bench.scaled_memories(base.memories, args.n)
    contents = [memory.content for memory in memories]
    started = time.perf_counter()
    lex_codes = [axo.sdr.encode_lexical(text) for text in contents]
    encode_s = time.perf_counter() - started

    store = MemoryStore(":memory:")
    for memory, content, code in zip(memories, contents, lex_codes, strict=True):
        store.add(
            MemoryItem(
                memory_id=memory.memory_id,
                kind=MemoryKind.FACT,
                tier=Tier.RECENT,
                content=content,
                created_at=1_700_000_000.0,
                lex_sdr=code,
            )
        )
    started = time.perf_counter()
    index = store.lexical_index()
    build_s = time.perf_counter() - started
    lengths = np.zeros(len(index.ids), dtype=np.int64)
    by_id = {memory.memory_id: code for memory, code in zip(memories, lex_codes, strict=True)}
    for position, memory_id in enumerate(index.ids):
        lengths[position] = by_id[memory_id].k

    queries = base.queries
    limits = [int(value) for value in args.limits.split(",") if value.strip()]
    records = []
    for limit in limits:
        current_times = []
        exact_times = []
        coverage = []
        agree = []
        current_metrics = []
        exact_metrics = []
        candidate_sizes = []
        for query in queries:
            query_sdr = axo.sdr.encode_lexical(query.query)

            started = time.perf_counter()
            current_ids = index.candidates(query_sdr, limit=limit)
            current_times.append(time.perf_counter() - started)

            started = time.perf_counter()
            exact_rows = ranked_exact(index, query_sdr, lengths, limit)
            exact_times.append(time.perf_counter() - started)
            exact_ids = [index.ids[int(row)] for row in exact_rows]

            full_rows = ranked_exact(index, query_sdr, lengths, 8)
            full_ids = [index.ids[int(row)] for row in full_rows]
            coverage.append(len(set(full_ids) & set(current_ids)) / max(1, len(full_ids)))
            agree.append(1.0 if current_ids and exact_ids and current_ids[0] == exact_ids[0] else 0.0)
            candidate_sizes.append(len(current_ids))
            current_metrics.append(metrics(list(current_ids[:8]), query.expect))
            exact_metrics.append(metrics(exact_ids[:8], query.expect))
        records.append(
            {
                "limit": limit,
                "n": len(index.ids),
                "build_s": build_s,
                "current_ms": float(np.mean(current_times) * 1000),
                "exact_ms": float(np.mean(exact_times) * 1000),
                "candidates": float(np.mean(candidate_sizes)),
                "top8_coverage": mean(coverage),
                "top1_agree": mean(agree),
                "current_hit1": mean([row["hit1"] for row in current_metrics]),
                "current_hit8": mean([row["hit8"] for row in current_metrics]),
                "current_mrr": mean([row["mrr"] for row in current_metrics]),
                "exact_hit1": mean([row["hit1"] for row in exact_metrics]),
                "exact_hit8": mean([row["hit8"] for row in exact_metrics]),
                "exact_mrr": mean([row["mrr"] for row in exact_metrics]),
                "queries": len(queries),
            }
        )
        row = records[-1]
        print(
            f"limit={limit:<6} cand {row['candidates']:.0f} "
            f"current {row['current_ms']:.1f}ms hit1 {row['current_hit1']:.3f} "
            f"hit8 {row['current_hit8']:.3f} | exact {row['exact_ms']:.1f}ms "
            f"hit1 {row['exact_hit1']:.3f} hit8 {row['exact_hit8']:.3f} "
            f"cover {row['top8_coverage']:.3f} agree {row['top1_agree']:.3f}"
        )

    payload = {
        "n": args.n,
        "encode_s": encode_s,
        "build_s": build_s,
        "queries": len(queries),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
