"""Axo 关联基质的神经元式选位探针。

预注册：[`sweep_results/axo_substrate_prereg.md`](../sweep_results/axo_substrate_prereg.md)。

**它不在 Axo 仓里改任何东西**：只把 Axo 的纯层（``memcore`` + ``axo.bench.substrate``）
按路径导入，读它冻结的语料，用它的指标函数算数。所有新代码留在 BioSNN 的研究副本里。

用法::

    uv run python scripts/axo_substrate_probe.py --baseline
    uv run python scripts/axo_substrate_probe.py            # 默认网格，k=64
    uv run python scripts/axo_substrate_probe.py --arms idf-1.0,homeo-0.5 --k 32,128,256
    uv run python scripts/axo_substrate_probe.py --scale 500,1000,2000 --arms idf-1.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

AXO_SRC_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/src")
AXO_CORPUS_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/benchmarks/substrate")

#: 预注册里写死的判据。
A_BASELINE = 0.7083
W_BASELINE = 0.8700
W_BEST = 0.9000
LEXICAL_HIT1 = 0.8500
LEXICAL_MRR = 0.9021
ORPHAN_ALARM = 100


class Axo:
    """按路径导入 Axo 的纯层。**不安装、不修改那个仓。**"""

    def __init__(self, src: Path) -> None:
        sys.path.insert(0, str(src))
        from axo.bench import substrate as bench
        from memcore import sdr, types

        self.bench = bench
        self.sdr = sdr
        self.SDR = types.SDR
        self.width = types.DEFAULT_SDR_WIDTH

    def load(self, corpus_dir: Path):
        return self.bench.load_substrate_corpus(corpus_dir)


def lexical_codes(corpus, axo: Axo):
    """一次算好词法码，所有臂共用。"""
    return tuple(axo.sdr.encode_lexical(memory.content) for memory in corpus.memories)


def document_frequency(lex_codes, width: int) -> np.ndarray:
    """每个位在多少条记忆里出现。**语料级统计，不是随机数。**"""
    counts = np.zeros(width, dtype=np.int64)
    for sdr in lex_codes:
        counts[sdr.idx] += 1
    return counts


def _select(lex, scores: np.ndarray, k: int, axo: Axo):
    """按分数取前 k 位，保留原词法权重，下标升序。"""
    positions = np.lexsort((lex.idx, -scores))[: min(k, lex.k)]
    positions = positions[np.argsort(lex.idx[positions], kind="stable")]
    return axo.SDR(
        idx=lex.idx[positions].astype(np.int32),
        weights=lex.weights[positions],
        width=lex.width,
    )


def _hybrid(lex, df: np.ndarray, n: int, k: int, hash_bits: int, axo: Axo):
    """``hash_bits`` 位走 Axo 的 hash 选位，其余按 ``idf-2.0`` 选。

    混合的理由：hash 选位保住词法 Jaccard（图 A），idf 选位压掉跨条目公共位
    （矩阵 W）。两半各自解决一个失效模式，而不是让同一条规则同时负责两边。
    """
    hash_code = axo.sdr.encode_assoc(lex, k=min(hash_bits, k), mode="hash")
    selected = {int(i) for i in hash_code.idx}
    frequencies = df[lex.idx].astype(np.float64)
    idf = np.log((n + 1.0) / (frequencies + 1.0)) + 1.0
    scores = lex.weights * (idf**2.0)
    order = np.lexsort((lex.idx, -scores))
    chosen = [int(position) for position in order if int(lex.idx[position]) not in selected]
    chosen = chosen[: max(k - len(selected), 0)]
    positions = np.concatenate(
        [np.searchsorted(lex.idx, hash_code.idx), np.asarray(chosen, dtype=np.int64)]
    )
    positions = np.unique(positions)
    return axo.SDR(
        idx=lex.idx[positions].astype(np.int32),
        weights=lex.weights[positions],
        width=lex.width,
    )


def encode_arm(arm: str, lex_codes, df: np.ndarray, n: int, k: int, axo: Axo):
    """一个臂的关联码。``hash``/``weight`` 走 Axo 原实现，其余是选位分数。"""
    if arm == "hash":
        return tuple(axo.sdr.encode_assoc(lex, k=k, mode="hash") for lex in lex_codes)
    if arm == "weight":
        return tuple(axo.sdr.encode_assoc(lex, k=k, mode="weight") for lex in lex_codes)

    family, _, value = arm.partition("-")
    strength = float(value)
    if family == "hybrid":
        return tuple(_hybrid(lex, df, n, k, int(strength), axo) for lex in lex_codes)
    if family not in ("idf", "homeo"):
        raise SystemExit(
            f"不认识的臂 {arm!r}；只认得 hash / weight / idf-<γ> / homeo-<β> / hybrid-<h>。"
        )

    codes = []
    for lex in lex_codes:
        frequencies = df[lex.idx].astype(np.float64)
        if family == "idf":
            idf = np.log((n + 1.0) / (frequencies + 1.0)) + 1.0
            scores = lex.weights * (idf**strength)
        else:
            scores = lex.weights - strength * (frequencies / n)
        codes.append(_select(lex, scores, k, axo))
    return tuple(codes)


def code_statistics(codes, clusters) -> dict[str, float]:
    """有效维数与簇内/簇外 Jaccard——诊断用的，不进判据。"""
    width = max(int(code.width) for code in codes)
    dense = np.zeros((len(codes), width), dtype=np.float64)
    for row, code in enumerate(codes):
        dense[row, code.idx] = 1.0
    cardinality = dense.sum(axis=1)
    intersection = dense @ dense.T
    union = cardinality[:, None] + cardinality[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, intersection / union, 0.0)

    same = np.array(
        [clusters[i] == clusters[j] for i in range(len(codes)) for j in range(i + 1, len(codes))]
    )
    values = np.array([jaccard[i, j] for i in range(len(codes)) for j in range(i + 1, len(codes))])
    union_bits = int(np.count_nonzero(dense.any(axis=0)))
    return {
        "union_bits": float(union_bits),
        "mean_jaccard": float(values.mean()) if values.size else float("nan"),
        "mean_jaccard_same": float(values[same].mean()) if same.any() else float("nan"),
        "mean_jaccard_diff": float(values[~same].mean()) if (~same).any() else float("nan"),
    }


def evaluate(
    coded,
    corpus,
    axo: Axo,
    *,
    drop: float,
    top_m: int,
    min_weight: float,
    decay: float,
) -> dict:
    """用 Axo 自己的指标函数算一项臂。"""
    bench = axo.bench
    cluster_of = {memory.memory_id: memory.cluster for memory in corpus.memories}
    graph = bench.build_graph(coded, top_m=top_m, min_weight=min_weight, k=64, mode="hash")
    pairs = bench.neighbor_pairs(graph, cluster_of)
    probes = bench.probe_sample(corpus.memories, 200)
    outcomes = bench.completion_outcomes(
        coded, [memory.memory_id for memory in probes], drop=drop, top=5
    )
    graph_outcomes = bench.graph_outcomes(coded, corpus.queries, graph, decay=decay, depth=10)

    metrics = {}
    metrics.update(bench.neighbor_summary(pairs))
    metrics.update(bench.completion_summary(outcomes))
    metrics.update(bench.graph_summary(graph_outcomes))
    clusters = [memory.cluster for memory in corpus.memories]
    metrics.update(code_statistics([item.assoc_sdr for item in coded], clusters))
    return metrics


def run_arm(
    arm: str,
    corpus,
    lex_codes,
    df: np.ndarray,
    axo: Axo,
    *,
    k: int,
    drop: float,
    top_m: int,
    min_weight: float,
    decay: float,
) -> dict:
    started = time.perf_counter()
    codes = encode_arm(arm, lex_codes, df, len(corpus.memories), k, axo)
    elapsed = time.perf_counter() - started
    coded = tuple(
        axo.bench.CodedMemory(
            memory_id=memory.memory_id,
            cluster=memory.cluster,
            lex_sdr=lex,
            assoc_sdr=code,
        )
        for memory, lex, code in zip(corpus.memories, lex_codes, codes, strict=True)
    )
    metrics = evaluate(
        coded, corpus, axo, drop=drop, top_m=top_m, min_weight=min_weight, decay=decay
    )
    return {
        "arm": arm,
        "k": k,
        "n": len(corpus.memories),
        "elapsed_encode_s": elapsed,
        **metrics,
    }


def qualified(record: dict) -> bool:
    """预注册的主判据。"""
    return (
        record["neighbor_auc"] >= A_BASELINE
        and record["completion_top1"] >= W_BASELINE
        and record["orphan_nodes"] <= ORPHAN_ALARM
    )


def print_row(record: dict) -> None:
    print(
        f"{record['arm']:<12} k={record['k']:<4} "
        f"AUC {record['neighbor_auc']:.4f}  "
        f"top1 {record['completion_top1']:.4f}  "
        f"top5 {record['completion_top5']:.4f}  "
        f"图hit@1 {record['graph_hit@1']:.4f}  "
        f"图MRR {record['graph_mrr']:.4f}  "
        f"并集位 {record['union_bits']:.0f}  "
        f"同簇J {record['mean_jaccard_same']:.4f}  "
        f"异簇J {record['mean_jaccard_diff']:.4f}  "
        f"孤点 {record['orphan_nodes']:.0f}  "
        f"{record['elapsed_encode_s']:.2f}s"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--corpus-dir", type=Path, default=AXO_CORPUS_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=Path("sweep_results/axo_substrate_probe"))
    parser.add_argument("--baseline", action="store_true", help="只复现 hash / weight 两条基线")
    parser.add_argument(
        "--arms",
        default="hash,weight,idf-0.5,idf-1.0,idf-2.0,homeo-0.25,homeo-0.5,homeo-1.0",
        help="逗号分隔的臂名",
    )
    parser.add_argument("--k", default="64", help="逗号分隔的活跃位数")
    parser.add_argument("--scale", default="", help="逗号分隔的语料规模；空 = 原语料")
    parser.add_argument("--drop", type=float, default=0.5)
    parser.add_argument("--top-m", type=int, default=8)
    parser.add_argument("--min-weight", type=float, default=0.10)
    parser.add_argument("--decay", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    axo = Axo(args.axo_src)
    base_corpus = axo.load(args.corpus_dir)
    digests = axo.bench.substrate_digests(args.corpus_dir)

    arms = (
        ["hash", "weight"]
        if args.baseline
        else [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    )
    ks = [int(value) for value in args.k.split(",") if value.strip()]
    scales = [int(value) for value in args.scale.split(",") if value.strip()] or [
        len(base_corpus.memories)
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for scale in scales:
        corpus_memories = axo.bench.scaled_memories(base_corpus.memories, scale)
        corpus = axo.bench.SubstrateCorpus(
            memories=tuple(corpus_memories),
            queries=base_corpus.queries,
            corpus_dir=base_corpus.corpus_dir,
        )
        lex_codes = lexical_codes(corpus, axo)
        df = document_frequency(lex_codes, axo.width)
        for k in ks:
            for arm in arms:
                record = run_arm(
                    arm,
                    corpus,
                    lex_codes,
                    df,
                    axo,
                    k=k,
                    drop=args.drop,
                    top_m=args.top_m,
                    min_weight=args.min_weight,
                    decay=args.decay,
                )
                record["corpus_sha256"] = digests["memories"]["sha256"]
                record["scale"] = scale
                record["qualified"] = qualified(record)
                records.append(record)
                print_row(record)

    payload = {"corpus": digests, "records": records}
    out_path = args.out_dir / ("baseline.json" if args.baseline else "grid.json")
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[落盘] {out_path}")

    winners = [record for record in records if record["qualified"]]
    if winners:
        print("\n合格臂（同时 ≥ A 基线 与 W 基线）：")
        for record in winners:
            print_row(record)
    else:
        print("\n没有臂同时达到 A 基线与 W 基线。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
