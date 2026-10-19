"""50k 增量词法索引：新增记忆不再整库重建。

量三件事：
1. 首次整库索引构建；
2. 往已建好索引的 50k 库里追加 100 条、再追加到 delta 上限的耗时；
3. delta 满后下一次 `lexical_index()` 的全量合并耗时，以及追加期间的召回耗时。

语料仍是 `benchmarks/substrate` 的真实文本放大；只量索引与召回，不量语义质量。
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
    parser.add_argument("--first-batch", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("sweep_results/axo_recall_50k/incremental.json"))
    args = parser.parse_args(argv)

    axo = probe.Axo(args.axo_src)
    import memcore.store as store_module
    from memcore.recall import RecallConfig, recall
    from memcore.store import MemoryStore
    from memcore.types import MemoryItem, MemoryKind, Tier

    base = axo.load(args.corpus_dir)
    memories = axo.bench.scaled_memories(base.memories, args.n)
    contents = [memory.content for memory in memories]
    started = time.perf_counter()
    codes = [axo.sdr.encode_lexical(text) for text in contents]
    encode_s = time.perf_counter() - started

    store = MemoryStore(":memory:")
    started = time.perf_counter()
    for memory, content, code in zip(memories, contents, codes, strict=True):
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
    add_s = time.perf_counter() - started

    started = time.perf_counter()
    store.lexical_index()
    build_s = time.perf_counter() - started

    new_contents = [f"新增记忆{index}：" + contents[index % len(contents)] for index in range(600)]
    new_codes = [axo.sdr.encode_lexical(text) for text in new_contents]

    def add_batch(start: int, stop: int) -> float:
        started = time.perf_counter()
        for index in range(start, stop):
            store.add(
                MemoryItem(
                    memory_id=f"z{index:06d}",
                    kind=MemoryKind.FACT,
                    tier=Tier.RECENT,
                    content=new_contents[index],
                    created_at=1_700_100_000.0 + index,
                    lex_sdr=new_codes[index],
                )
            )
        return time.perf_counter() - started

    def timed_recall() -> tuple[float, int]:
        now = 1_700_200_000.0
        started = time.perf_counter()
        ledger = recall(
            store,
            query_text=base.queries[0].query,
            budget_tokens=10**9,
            config=RecallConfig(),
            now=now,
            settle=False,
        )
        return time.perf_counter() - started, ledger.scanned

    first_add_s = add_batch(0, args.first_batch)
    cached = store.lexical_index()
    cached_delta = len(cached.delta_ids)
    recall_ms, scanned = timed_recall()

    remaining = max(0, store_module._LEX_DELTA_MAX - args.first_batch)
    second_add_s = add_batch(args.first_batch, args.first_batch + remaining)
    before_merge = store.lexical_index()
    before_merge_delta = len(before_merge.delta_ids)

    # 再写一条就会越过 delta 上限；下一次 lexical_index() 做全量合并。
    add_batch(args.first_batch + remaining, args.first_batch + remaining + 1)
    started = time.perf_counter()
    merged = store.lexical_index()
    merge_s = time.perf_counter() - started
    merged_delta = len(merged.delta_ids)
    merged_recall_ms, merged_scanned = timed_recall()

    payload = {
        "n": args.n,
        "encode_s": encode_s,
        "add_s": add_s,
        "build_s": build_s,
        "delta_limit": store_module._LEX_DELTA_MAX,
        "first_batch": args.first_batch,
        "first_add_s": first_add_s,
        "first_add_ms": first_add_s / args.first_batch * 1000,
        "cached_delta": cached_delta,
        "recall_ms": recall_ms * 1000,
        "scanned": scanned,
        "second_add_s": second_add_s,
        "second_add_ms": (second_add_s / remaining * 1000) if remaining else None,
        "before_merge_delta": before_merge_delta,
        "merge_s": merge_s,
        "merged_delta": merged_delta,
        "merged_recall_ms": merged_recall_ms * 1000,
        "merged_scanned": merged_scanned,
        "total": len(contents) + args.first_batch + remaining + 1,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(payload)
    print(f"[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
