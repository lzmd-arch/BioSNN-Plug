"""50k 召回：词法 + 语义并集候选的 profiling。

给 50k 条真实文本记忆加**确定性合成 768 维向量**（不是 nomic 的真实嵌入；只量
通道耗时与候选并集，不量语义质量），跑真正的 `recall`。
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
    parser.add_argument("--queries", type=int, default=5)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("sweep_results/axo_recall_50k/semantic.json"),
    )
    args = parser.parse_args(argv)

    axo = probe.Axo(args.axo_src)
    from memcore.recall import RecallConfig, recall
    from memcore.spaces import Space, make_space_id
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

    space = Space(space_id=make_space_id("test", "random", args.dim), dim=args.dim)
    store.register_space(space)
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((len(contents), args.dim), dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    started = time.perf_counter()
    rows = [
        (
            f"m{index:06d}",
            space.space_id,
            args.dim,
            matrix[index].tobytes(),
            1_700_000_000.0 + index,
        )
        for index in range(len(contents))
    ]
    with store._lock:
        store._connection.executemany(
            "INSERT INTO embeddings (memory_id, space_id, dim, vec, added_at) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        store._connection.commit()
    embedding_s = time.perf_counter() - started

    items = list(store.iter_items())
    started = time.perf_counter()
    store.lexical_index()
    lexical_index_s = time.perf_counter() - started
    now = max(item.created_at for item in items) + 1.0
    indices = np.unique(
        np.linspace(0, len(items) - 1, min(args.queries, len(items))).round().astype(int)
    )
    recall_times = []
    scanned = []
    for index in indices:
        started = time.perf_counter()
        ledger = recall(
            store,
            query_text=items[int(index)].content,
            query_vector=matrix[int(index)],
            space_id=space.space_id,
            budget_tokens=10**9,
            config=RecallConfig(),
            now=now,
            settle=False,
        )
        recall_times.append(time.perf_counter() - started)
        scanned.append(ledger.scanned)

    payload = {
        "n": len(items),
        "dim": args.dim,
        "encode_s": encode_s,
        "add_s": add_s,
        "embedding_s": embedding_s,
        "lexical_index_s": lexical_index_s,
        "recall_ms": float(np.mean(recall_times) * 1000),
        "recall_median_ms": float(np.median(recall_times) * 1000),
        "recall_first_ms": float(recall_times[0] * 1000),
        "recall_warm_ms": float(np.mean(recall_times[1:]) * 1000)
        if len(recall_times) > 1
        else None,
        "scanned": int(np.mean(scanned)),
        "queries": len(indices),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(payload)
    print(f"[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
