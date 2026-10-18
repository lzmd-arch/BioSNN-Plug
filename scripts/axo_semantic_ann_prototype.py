"""语义 ANN 候选预取原型（LSH）。

**只做候选**：最终打分/排序不在这里，仍归召回/SNN 基质。目标是量“精确 top-k
邻居有多少落进 ANN 候选”，召回不够就不引入。

数据：真实 166 条 nomic 嵌入（只读）+ 50k 确定性合成 768 维。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

AXO_DATA_DEFAULT = Path("D:/AxoData")


def load_real_embeddings(data_root: Path) -> np.ndarray:
    path = data_root / "memory" / "mem.db"
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        rows = connection.execute("SELECT dim, vec FROM embeddings ORDER BY memory_id").fetchall()
    finally:
        connection.close()
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    dim = int(rows[0][0])
    matrix = np.vstack([np.frombuffer(bytes(vec), dtype=np.float32, count=dim) for _, vec in rows])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def synthetic_embeddings(n: int, dim: int) -> np.ndarray:
    generator = np.random.default_rng(0)
    matrix = generator.standard_normal((n, dim), dtype=np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return matrix


def lsh_candidates(
    matrix: np.ndarray,
    *,
    tables: int,
    bits: int,
    top_k: int,
    queries: int,
) -> dict:
    generator = np.random.default_rng(1)
    planes = generator.standard_normal((matrix.shape[1], tables * bits), dtype=np.float32)
    started = time.perf_counter()
    projections = matrix @ planes
    signs = projections > 0
    project_s = time.perf_counter() - started
    powers = 1 << np.arange(bits, dtype=np.int64)
    keys = np.empty((matrix.shape[0], tables), dtype=np.int64)
    for table in range(tables):
        block = signs[:, table * bits : (table + 1) * bits]
        keys[:, table] = block.astype(np.int64) @ powers
    buckets = [{key: [] for key in np.unique(keys[:, table])} for table in range(tables)]
    for row in range(matrix.shape[0]):
        for table in range(tables):
            buckets[table][int(keys[row, table])].append(row)

    indices = np.unique(
        np.linspace(0, matrix.shape[0] - 1, min(queries, matrix.shape[0])).round().astype(int)
    )
    recalls = []
    candidate_sizes = []
    query_times = []
    for index in indices:
        started = time.perf_counter()
        gathered = set()
        for table in range(tables):
            gathered.update(buckets[table].get(int(keys[index, table]), ()))
        gathered.discard(int(index))
        query_times.append(time.perf_counter() - started)
        candidate_sizes.append(len(gathered))
        scores = matrix @ matrix[index]
        scores[index] = -np.inf
        exact = np.argsort(-scores, kind="stable")[:top_k]
        recalls.append(sum(1 for row in exact if int(row) in gathered) / top_k)
    return {
        "n": int(matrix.shape[0]),
        "tables": tables,
        "bits": bits,
        "top_k": top_k,
        "project_s": project_s,
        "recall_mean": float(np.mean(recalls)),
        "recall_min": float(np.min(recalls)),
        "candidates_mean": float(np.mean(candidate_sizes)),
        "query_ms": float(np.mean(query_times) * 1000),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=AXO_DATA_DEFAULT)
    parser.add_argument("--synthetic-n", type=int, default=50000)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument(
        "--out", type=Path, default=Path("sweep_results/axo_semantic_ann/scan.json")
    )
    args = parser.parse_args(argv)

    real = load_real_embeddings(args.data_root)
    synthetic = synthetic_embeddings(args.synthetic_n, args.dim)
    rows = []
    for name, matrix in (("real-166", real), ("synthetic-50k", synthetic)):
        if matrix.size == 0:
            continue
        for tables, bits in ((8, 6), (16, 6), (16, 8)):
            for top_k in (10, 50, 100):
                row = lsh_candidates(
                    matrix,
                    tables=tables,
                    bits=bits,
                    top_k=top_k,
                    queries=args.queries,
                )
                row["dataset"] = name
                rows.append(row)
                print(
                    f"{name} tables={tables} bits={bits} top{top_k}: "
                    f"recall {row['recall_mean']:.4f} (min {row['recall_min']:.4f}) "
                    f"cand {row['candidates_mean']:.0f} q {row['query_ms']:.2f}ms "
                    f"project {row['project_s']:.1f}s"
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"records": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
