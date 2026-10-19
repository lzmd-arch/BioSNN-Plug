"""50k 图通道：种子分从全库解码改为词法索引。

量两件事：
1. 旧实现 `iter_items()` 解码全库 SDR 取种子 vs 新实现 `LexicalIndex.scores()`；
2. 扩散后正分节点数——它决定 `_prefilter_ids` 的并集会不会超过 20,000 而回退全量。

图用**链式 stub**（每个节点一条出边），只量通道成本与正分规模，不量图质量；
建一张真 50k 图要 259 s，不在这一轮里做。
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


class ChainGraph:
    """每个节点一条出边到下一个节点；扩散取 max。"""

    def __init__(self, node_ids: tuple[str, ...]) -> None:
        self.node_ids = node_ids

    def __len__(self) -> int:
        return len(self.node_ids)

    def diffuse(self, base: np.ndarray, *, decay: float = 0.5) -> np.ndarray:
        scores = np.asarray(base, dtype=np.float64)
        spread = np.zeros(len(self.node_ids), dtype=np.float64)
        np.maximum.at(spread, (np.arange(len(self.node_ids)) + 1) % len(self.node_ids), scores * decay)
        return spread


def old_seeds(store, query_lex, graph):
    lex_of = {
        item.memory_id: item.lex_sdr
        for item in store.iter_items()
        if item.lex_sdr is not None
    }
    base = np.zeros(len(graph), dtype=np.float64)
    for row, node_id in enumerate(graph.node_ids):
        source = lex_of.get(node_id)
        if source is None or source.width != query_lex.width:
            continue
        base[row] = source.jaccard(query_lex)
    return base


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DEFAULT)
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--out", type=Path, default=Path("sweep_results/axo_recall_50k/graph_channel.json"))
    args = parser.parse_args(argv)

    axo = probe.Axo(args.axo_src)
    from memcore.recall import RecallConfig, _graph_channel, _prefilter_ids, recall
    from memcore.store import MemoryStore
    from memcore.types import MemoryItem, MemoryKind, Tier

    base = axo.load(args.corpus_dir)
    memories = axo.bench.scaled_memories(base.memories, args.n)
    contents = [memory.content for memory in memories]
    started = time.perf_counter()
    codes = [axo.sdr.encode_lexical(text) for text in contents]
    encode_s = time.perf_counter() - started

    store = MemoryStore(":memory:")
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
    started = time.perf_counter()
    store.lexical_index()
    build_s = time.perf_counter() - started

    graph = ChainGraph(tuple(memory.memory_id for memory in memories))
    query_text = base.queries[0].query
    query = axo.sdr.encode_lexical(query_text)

    started = time.perf_counter()
    old_base = old_seeds(store, query, graph)
    old_seed_s = time.perf_counter() - started

    started = time.perf_counter()
    new_base = np.zeros(len(graph), dtype=np.float64)
    lex_scores = store.lexical_index().scores(query)
    for row, node_id in enumerate(graph.node_ids):
        new_base[row] = lex_scores.get(node_id, 0.0)
    new_seed_s = time.perf_counter() - started
    assert np.array_equal(old_base, new_base)

    started = time.perf_counter()
    scores = _graph_channel(store, query_lex=query, graph=graph, decay=0.5)
    channel_s = time.perf_counter() - started

    started = time.perf_counter()
    candidate_ids = _prefilter_ids(
        store,
        query_lex=query,
        scores_by_id=None,
        graph_scores=scores,
        window_ids=None,
    )
    prefilter_s = time.perf_counter() - started
    union_size = None if candidate_ids is None else len(candidate_ids)

    started = time.perf_counter()
    ledger = recall(
        store,
        query_text=query_text,
        budget_tokens=10**9,
        config=RecallConfig(graph_weight=0.5),
        graph=graph,
        now=1_700_000_000.0 + args.n + 1.0,
        settle=False,
    )
    graph_recall_s = time.perf_counter() - started

    payload = {
        "n": args.n,
        "encode_s": encode_s,
        "build_s": build_s,
        "old_seed_s": old_seed_s,
        "new_seed_s": new_seed_s,
        "channel_s": channel_s,
        "seed_positive": int(np.count_nonzero(old_base)),
        "spread_positive": len(scores),
        "prefilter_s": prefilter_s,
        "union_size": union_size,
        "graph_recall_ms": graph_recall_s * 1000,
        "graph_scanned": ledger.scanned,
        "candidate_floor": 1000,
        "union_limit": 20000,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(payload)
    print(f"[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
