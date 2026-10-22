"""真 50k 关联图：图正分截到 5,000 与不截断的质量对照。

建一张真的 `GraphIndex`（不是 stub），用 substrate 的查询与 expect 集，比较：
- `_GRAPH_CANDIDATE_LIMIT = 5000`（截断，候选并集可扫）；
- `_GRAPH_CANDIDATE_LIMIT = 10**9`（不截断，正分 50k → 并集超限 → 全量扫描）。

指标 hit@1 / hit@8 / MRR / 注入序列一致率；两次都用同一张图、同一批查询、同一个
`now`。只量图通道的质量代价，不改 Axo 默认值。
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


def metrics(injected: list[str], expect: tuple[str, ...]) -> tuple[float, float, float]:
    expected = set(expect)
    if not expected:
        return None, None, None
    first = next(
        (position for position, memory_id in enumerate(injected, start=1) if memory_id in expected),
        None,
    )
    return (
        1.0 if injected and injected[0] in expected else 0.0,
        1.0 if first is not None and first <= 8 else 0.0,
        0.0 if first is None else 1.0 / first,
    )


def mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DEFAULT)
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--weights", default="0.25,1.0")
    parser.add_argument(
        "--out", type=Path, default=Path("sweep_results/axo_recall_50k/graph_quality.json")
    )
    args = parser.parse_args(argv)

    axo = probe.Axo(args.axo_src)
    import memcore.recall as recall_module
    from memcore.graph import GraphIndex
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
    items = list(store.iter_items())

    started = time.perf_counter()
    graph = GraphIndex.build(items)
    graph_s = time.perf_counter() - started

    indices = np.unique(
        np.linspace(0, len(base.queries) - 1, min(args.queries, len(base.queries)))
        .round()
        .astype(int)
    )
    queries = [base.queries[int(index)] for index in indices]
    now = 1_700_000_000.0 + args.n + 1.0
    original_limit = recall_module._GRAPH_CANDIDATE_LIMIT

    records = []
    try:
        for weight in [float(value) for value in args.weights.split(",") if value.strip()]:
            runs = {}
            for label, limit in (("capped", original_limit), ("uncapped", 10**9)):
                recall_module._GRAPH_CANDIDATE_LIMIT = limit
                injected_runs = []
                times = []
                scanned = []
                hit1 = []
                hit8 = []
                mrr = []
                for query in queries:
                    started = time.perf_counter()
                    ledger = recall(
                        store,
                        query_text=query.query,
                        budget_tokens=10**9,
                        config=RecallConfig(graph_weight=weight),
                        graph=graph,
                        now=now,
                        settle=False,
                    )
                    times.append(time.perf_counter() - started)
                    scanned.append(ledger.scanned)
                    injected = ledger.injected_ids
                    injected_runs.append(injected)
                    one, eight, reciprocal = metrics(injected, query.expect)
                    hit1.append(one)
                    hit8.append(eight)
                    mrr.append(reciprocal)
                runs[label] = {
                    "injected": injected_runs,
                    "hit1": mean(hit1),
                    "hit8": mean(hit8),
                    "mrr": mean(mrr),
                    "mean_ms": float(np.mean(times) * 1000),
                    "scanned": float(np.mean(scanned)),
                }
            capped = runs["capped"]
            uncapped = runs["uncapped"]
            top1_same = np.mean(
                [
                    (a[:1] == b[:1])
                    for a, b in zip(capped["injected"], uncapped["injected"], strict=True)
                ]
            )
            top8_same = np.mean(
                [
                    (a[:8] == b[:8])
                    for a, b in zip(capped["injected"], uncapped["injected"], strict=True)
                ]
            )
            records.append(
                {
                    "weight": weight,
                    "queries": len(queries),
                    "capped": {k: v for k, v in capped.items() if k != "injected"},
                    "uncapped": {k: v for k, v in uncapped.items() if k != "injected"},
                    "top1_same": float(top1_same),
                    "top8_same": float(top8_same),
                }
            )
            print(
                f"w={weight} capped hit1={capped['hit1']:.4f} hit8={capped['hit8']:.4f} "
                f"MRR={capped['mrr']:.4f} {capped['mean_ms']:.0f}ms scan={capped['scanned']:.0f} | "
                f"uncapped hit1={uncapped['hit1']:.4f} hit8={uncapped['hit8']:.4f} "
                f"MRR={uncapped['mrr']:.4f} {uncapped['mean_ms']:.0f}ms scan={uncapped['scanned']:.0f} | "
                f"top1 same={top1_same:.3f} top8 same={top8_same:.3f}",
                flush=True,
            )
    finally:
        recall_module._GRAPH_CANDIDATE_LIMIT = original_limit

    payload = {
        "n": args.n,
        "encode_s": encode_s,
        "add_s": add_s,
        "graph_s": graph_s,
        "graph_nodes": len(graph),
        "graph_edges": graph.edge_count(),
        "top_m": graph.top_m,
        "min_weight": graph.min_weight,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
