"""真实文本 50k：AISHELL-3 + 真实 nomic 嵌入的语义召回。

与 `axo_real_50k_lexical.py` 用同一份确定性抽样与查询（字符丢 30%），补上本机
Ollama `nomic-embed-text` 的 768 维真实向量，量语义并集候选、warm 召回与
候选/全量一致率。

**不是个人记忆库**：AISHELL-3 是公开语音转写，中位 11 字。语料不入库，只登记
路径与哈希。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import axo_real_50k_lexical as real
import axo_substrate_probe as probe

AXO_SRC_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/src")
AISHELL_DEFAULT = Path(r"D:/Projects/_scratch_ui/aishell3_full.jsonl")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--aishell", type=Path, default=AISHELL_DEFAULT)
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--min-chars", type=int, default=6)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--full-scan-queries", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--out", type=Path, default=Path("sweep_results/axo_real_50k/semantic.json")
    )
    args = parser.parse_args(argv)

    axo = probe.Axo(args.axo_src)
    import memcore.recall as recall_module
    from axo.llm.ollama import OllamaEmbeddingBackend
    from memcore.recall import RecallConfig, recall
    from memcore.spaces import Space, make_space_id
    from memcore.store import MemoryStore
    from memcore.types import MemoryItem, MemoryKind, Tier

    started = time.perf_counter()
    texts, source_digest = real.load_texts(args.aishell, min_chars=args.min_chars, n=args.n)
    load_s = time.perf_counter() - started

    started = time.perf_counter()
    codes = [axo.sdr.encode_lexical(text) for text in texts]
    encode_s = time.perf_counter() - started

    store = MemoryStore(":memory:")
    for index, (text, code) in enumerate(zip(texts, codes, strict=True)):
        store.add(
            MemoryItem(
                memory_id=f"a{index:06d}",
                kind=MemoryKind.FACT,
                tier=Tier.RECENT,
                content=text,
                created_at=1_700_000_000.0 + index,
                lex_sdr=code,
            )
        )
    store.lexical_index()

    backend = OllamaEmbeddingBackend(model="nomic-embed-text", dim=768, batch_size=args.batch_size)
    started = time.perf_counter()
    matrix = backend.embed(texts)
    embedding_s = time.perf_counter() - started
    if matrix.shape != (args.n, 768):
        raise SystemExit(f"嵌入形状 {matrix.shape}，期望 {(args.n, 768)}")

    space_id = make_space_id("ollama", "nomic-embed-text", 768)
    store.register_space(Space(space_id=space_id, dim=768, model_ref="ollama/nomic-embed-text"))
    rows = [
        (f"a{index:06d}", space_id, 768, matrix[index].tobytes(), 1_700_000_000.0)
        for index in range(args.n)
    ]
    started = time.perf_counter()
    with store._lock:
        store._connection.executemany(
            "INSERT INTO embeddings (memory_id, space_id, dim, vec, added_at) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        store._connection.commit()
        store._embedding_revision += 1
    insert_s = time.perf_counter() - started

    rng = np.random.default_rng(1)
    query_indices = np.sort(rng.choice(args.n, size=args.queries, replace=False))
    queries = [
        (f"a{int(index):06d}", real.perturb(texts[int(index)], rng)) for index in query_indices
    ]
    query_vectors = backend.embed([query for _, query in queries])
    now = 1_700_000_000.0 + args.n + 1.0

    hit1, hit8, mrr, times, scanned = [], [], [], [], []
    for (expected, query), vector in zip(queries, query_vectors, strict=True):
        started = time.perf_counter()
        ledger = recall(
            store,
            query_text=query,
            query_vector=vector,
            space_id=space_id,
            budget_tokens=10**9,
            config=RecallConfig(),
            now=now,
            settle=False,
        )
        times.append(time.perf_counter() - started)
        scanned.append(ledger.scanned)
        one, eight, reciprocal = real.metrics(ledger.injected_ids, expected)
        hit1.append(one)
        hit8.append(eight)
        mrr.append(reciprocal)

    original_prefilter = recall_module._prefilter_ids
    full = []
    try:
        recall_module._prefilter_ids = lambda *a, **k: None
        for (expected, query), vector in zip(
            queries[: args.full_scan_queries],
            query_vectors[: args.full_scan_queries],
            strict=True,
        ):
            started = time.perf_counter()
            ledger = recall(
                store,
                query_text=query,
                query_vector=vector,
                space_id=space_id,
                budget_tokens=10**9,
                config=RecallConfig(),
                now=now,
                settle=False,
            )
            full.append(
                (
                    tuple(ledger.injected_ids[:8]),
                    time.perf_counter() - started,
                    ledger.scanned,
                )
            )
    finally:
        recall_module._prefilter_ids = original_prefilter

    candidate = []
    for (expected, query), vector in zip(
        queries[: args.full_scan_queries],
        query_vectors[: args.full_scan_queries],
        strict=True,
    ):
        ledger = recall(
            store,
            query_text=query,
            query_vector=vector,
            space_id=space_id,
            budget_tokens=10**9,
            config=RecallConfig(),
            now=now,
            settle=False,
        )
        candidate.append(tuple(ledger.injected_ids[:8]))

    payload = {
        "source": str(args.aishell),
        "source_sha256": source_digest,
        "n": args.n,
        "embedding_model": "nomic-embed-text@768",
        "load_s": load_s,
        "encode_s": encode_s,
        "embedding_s": embedding_s,
        "embedding_ms_per_text": embedding_s / args.n * 1000,
        "insert_s": insert_s,
        "queries": len(queries),
        "hit1": real.mean(hit1),
        "hit8": real.mean(hit8),
        "mrr": real.mean(mrr),
        "recall_ms": float(np.mean(times) * 1000),
        "recall_warm_ms": float(np.mean(times[1:]) * 1000) if len(times) > 1 else None,
        "scanned": float(np.mean(scanned)),
        "full_scan_queries": len(full),
        "full_scan_ms": float(np.mean([item[1] for item in full]) * 1000),
        "full_scan_scanned": float(np.mean([item[2] for item in full])),
        "top1_same": real.mean(
            [1.0 if a[:1] == b[0][:1] else 0.0 for a, b in zip(candidate, full, strict=True)]
        ),
        "top8_same": real.mean(
            [1.0 if a[:8] == b[0] else 0.0 for a, b in zip(candidate, full, strict=True)]
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(payload)
    print(f"[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
