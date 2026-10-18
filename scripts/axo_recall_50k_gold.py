"""50k 真实文本放大语料上的召回金标准：候选上限的端到端影响。

用 substrate 的 40 条查询与 expect 集，跑真正的 `recall`（纯词法），比较不同
`_PREFILTER_LIMIT` 下的 hit@1/@8、MRR 与耗时。不改 Axo 的默认值。
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


def outcome(injected, expect):
    expect = set(expect)
    if not expect:
        return None, None, None
    first = next(
        (position for position, memory_id in enumerate(injected, start=1) if memory_id in expect),
        None,
    )
    return (
        1.0 if injected and injected[0] in expect else 0.0,
        1.0 if first is not None and first <= 8 else 0.0,
        0.0 if first is None else 1.0 / first,
    )


def mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DEFAULT)
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--limits", default="500,1000,2000,5000,1000000000")
    parser.add_argument("--out", type=Path, default=Path("sweep_results/axo_recall_50k/gold.json"))
    args = parser.parse_args(argv)

    axo = probe.Axo(args.axo_src)
    import memcore.recall as recall_mod
    from memcore.recall import RecallConfig, recall
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
    store.lexical_index()
    build_s = time.perf_counter() - started
    now = 1_700_000_000.0 + args.n + 1.0

    records = []
    for limit in [int(value) for value in args.limits.split(",") if value.strip()]:
        recall_mod._PREFILTER_LIMIT = limit
        hit1, hit8, mrr, times, scanned, injected_counts = [], [], [], [], [], []
        for query in base.queries:
            started = time.perf_counter()
            ledger = recall(
                store,
                query_text=query.query,
                budget_tokens=10**9,
                config=RecallConfig(),
                now=now,
                settle=False,
            )
            times.append(time.perf_counter() - started)
            one, eight, reciprocal = outcome(ledger.injected_ids, query.expect)
            hit1.append(one)
            hit8.append(eight)
            mrr.append(reciprocal)
            scanned.append(ledger.scanned)
            injected_counts.append(len(ledger.injected))
        record = {
            "limit": limit,
            "n": args.n,
            "build_s": build_s,
            "hit1": mean(hit1),
            "hit8": mean(hit8),
            "mrr": mean(mrr),
            "mean_ms": float(np.mean(times) * 1000),
            "median_ms": float(np.median(times) * 1000),
            "scanned": float(np.mean(scanned)),
            "injected": float(np.mean(injected_counts)),
            "queries": len(base.queries),
        }
        records.append(record)
        print(
            f"limit={limit:<10} hit1 {record['hit1']:.4f} hit8 {record['hit8']:.4f} "
            f"MRR {record['mrr']:.4f} {record['mean_ms']:.1f}ms "
            f"scan {record['scanned']:.0f} inj {record['injected']:.1f}"
        )

    payload = {"n": args.n, "encode_s": encode_s, "build_s": build_s, "records": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
