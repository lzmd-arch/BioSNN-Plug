"""P4c 分簇回放对照：现有 ``weight`` 码 vs 稳态 ALIF 的 ``idf`` 码。

**它量的是 Axo 真实应用里的那一步**：``memcore.consolidation.partition`` 分出的每一簇，
在簇自己的 W 上做 50% 丢位回放，看 top-1 是不是自己。

用法::

    uv run python scripts/axo_cluster_replay_probe.py --scale 200,500,1000,2000

``idf-global`` 用全语料的文档频率，``idf-cluster`` 用**簇内**文档频率。后者是
P4c 的候选集成方式：回放本来就在簇内建 W，频率统计也只在簇内算，不跨库。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

AXO_SRC_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/src")
AXO_CORPUS_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/benchmarks/substrate")


class Coded:
    """``Substrate.build`` 只需要这两个字段。"""

    def __init__(self, memory_id: str, code) -> None:
        self.memory_id = memory_id
        self.assoc_sdr = code
        self.lex_sdr = None


def load_probe_module():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import axo_substrate_probe as probe

    return probe


def recognition(memories, lex_codes, codes_by_memory, clusters, axo) -> tuple[int, int]:
    from memcore.sdr import drop_bits
    from memcore.substrate import Substrate, SubstrateConfig

    hits = total = 0
    for cluster in clusters:
        members = list(cluster)
        substrate = Substrate.build(
            [Coded(mid, codes_by_memory[mid]) for mid in members],
            config=SubstrateConfig(k=64, mode="hash"),
        )
        for mid in members:
            ranked = substrate.complete(drop_bits(codes_by_memory[mid], 0.5), top=1)
            hits += bool(ranked) and ranked[0][0] == mid
            total += 1
    return hits, total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--corpus-dir", type=Path, default=AXO_CORPUS_DEFAULT)
    parser.add_argument("--scale", default="200,500,1000,2000")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument(
        "--out", type=Path, default=Path("sweep_results/axo_cluster_replay/scan.json")
    )
    args = parser.parse_args(argv)

    probe = load_probe_module()
    axo = probe.Axo(args.axo_src)
    from memcore.consolidation import partition

    base = axo.load(args.corpus_dir)
    rows = []
    for n in [int(value) for value in args.scale.split(",") if value.strip()]:
        memories = axo.bench.scaled_memories(base.memories, n)
        corpus = axo.bench.SubstrateCorpus(
            memories=tuple(memories), queries=base.queries, corpus_dir=base.corpus_dir
        )
        lex = probe.lexical_codes(corpus, axo)
        df_all = probe.document_frequency(lex, axo.width)
        hash_codes = probe.encode_arm("hash", lex, df_all, n, args.k, axo)
        coded = tuple(
            axo.bench.CodedMemory(m.memory_id, m.cluster, lex_code, c)
            for m, lex_code, c in zip(memories, lex, hash_codes, strict=True)
        )
        graph = axo.bench.build_graph(coded, top_m=8, min_weight=0.1, k=args.k, mode="hash")
        clusters = partition(graph)
        index = {m.memory_id: i for i, m in enumerate(memories)}

        arms = {
            "weight": {
                mid: axo.sdr.encode_assoc(lex[index[mid]], k=args.k, mode="weight") for mid in index
            },
        }
        arms["idf-global"] = dict(
            zip(
                index,
                probe.encode_arm("idf-2.0", lex, df_all, n, args.k, axo),
                strict=True,
            )
        )
        idf_cluster = {}
        for cluster in clusters:
            lex_members = [lex[index[mid]] for mid in cluster]
            df_cluster = probe.document_frequency(lex_members, axo.width)
            codes = probe.encode_arm("idf-2.0", lex_members, df_cluster, len(cluster), args.k, axo)
            idf_cluster.update(dict(zip(cluster, codes, strict=True)))
        arms["idf-cluster"] = idf_cluster

        print(f"n={n} clusters={len(clusters)} covered={sum(len(c) for c in clusters)}")
        for arm, codes_by_memory in arms.items():
            started = time.perf_counter()
            hits, total = recognition(memories, lex, codes_by_memory, clusters, axo)
            elapsed = time.perf_counter() - started
            row = {
                "scale": n,
                "arm": arm,
                "clusters": len(clusters),
                "covered": sum(len(c) for c in clusters),
                "hits": hits,
                "total": total,
                "recognition_rate": hits / total if total else float("nan"),
                "elapsed_s": elapsed,
            }
            rows.append(row)
            print(
                f"  {arm:<12} recognition {row['recognition_rate']:.4f} "
                f"({hits}/{total}) {elapsed:.2f}s"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"records": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\n[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
