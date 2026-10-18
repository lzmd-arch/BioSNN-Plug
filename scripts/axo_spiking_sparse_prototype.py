"""Axo 真脉冲原型第二轮：稀疏表示 + 序列遍历。

预注册：[`sweep_results/axo_spiking_sparse_prereg.md`](../sweep_results/axo_spiking_sparse_prereg.md)。

- 稀疏码：`(n, k=64)` 的 `(idx, weights)`，不再物化 `(n, 4096)`。
- 序列遍历：只用相邻轮 `t -> t+1` 学 `W_seq`，查询目标是 `t+L`；对照是速率迭代 L 步。
- 真实库只读副本；不输出正文。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import axo_spiking_prototype as base

AXO_SRC_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/src")
DATA_ROOT_DEFAULT = Path("D:/AxoData")


@dataclass(frozen=True)
class SparseCodes:
    idx: np.ndarray
    weights: np.ndarray
    width: int

    @property
    def n(self) -> int:
        return int(self.idx.shape[0])

    def matvec(self, dense: np.ndarray) -> np.ndarray:
        return (self.weights * dense[self.idx]).sum(axis=1)

    def transpose_matvec(self, scores: np.ndarray) -> np.ndarray:
        return np.bincount(
            self.idx.ravel(),
            weights=(self.weights * scores[:, None]).ravel(),
            minlength=self.width,
        )

    def subset(self, indices) -> SparseCodes:
        return SparseCodes(self.idx[indices], self.weights[indices], self.width)


def sparse_from_codes(codes, width: int) -> SparseCodes:
    k = max(code.k for code in codes)
    idx = np.zeros((len(codes), k), dtype=np.int32)
    weights = np.zeros((len(codes), k), dtype=np.float32)
    for row, code in enumerate(codes):
        idx[row, : code.k] = code.idx
        weights[row, : code.k] = code.weights
    return SparseCodes(idx=idx, weights=weights, width=width)


def dense_from_sparse(codes: SparseCodes) -> np.ndarray:
    matrix = np.zeros((codes.n, codes.width), dtype=np.float32)
    rows = np.repeat(np.arange(codes.n), codes.idx.shape[1])
    matrix[rows, codes.idx.ravel()] = codes.weights.ravel()
    return matrix


def rate_static_sparse(codes: SparseCodes, probe: np.ndarray) -> np.ndarray:
    activation = codes.transpose_matvec(codes.matvec(probe))
    return np.argsort(-codes.matvec(activation), kind="stable")


def rate_static_dense(matrix: np.ndarray, probe: np.ndarray) -> np.ndarray:
    activation = matrix.T @ (matrix @ probe)
    return np.argsort(-(matrix @ activation), kind="stable")


def simulate_sparse(
    readout: SparseCodes,
    prev: SparseCodes,
    nxt: SparseCodes,
    probe: np.ndarray,
    *,
    steps: int,
    theta: float,
    beta: float,
    rho: float,
    hetero: float,
    readout_mode: str,
    inhibition: str,
    kwta_fraction: float,
    global_gain: float,
) -> tuple[np.ndarray, float, float]:
    width = readout.width
    initial = base.normalize(nxt.transpose_matvec(prev.matvec(probe)))
    if hetero > 0.0:
        tau = 20.0 * np.exp(hetero * np.sin(np.arange(width, dtype=np.float64) * 0.61803398875))
        alpha = np.exp(-1.0 / tau).astype(np.float32)
    else:
        alpha = np.float32(0.95)

    v = np.zeros(width, dtype=np.float32)
    z = np.zeros(width, dtype=np.float32)
    adaptation = np.zeros(width, dtype=np.float32)
    refractory = np.zeros(width, dtype=np.int32)
    spike_count = np.zeros(width, dtype=np.float32)
    total_spikes = 0.0
    kwta = max(1, int(kwta_fraction * width))

    for _ in range(steps):
        recurrent = base.normalize(nxt.transpose_matvec(prev.matvec(z)))
        threshold = theta + beta * adaptation
        current = initial + recurrent
        if inhibition == "global" and global_gain > 0.0:
            current = current - global_gain * float(z.sum()) / max(1.0, kwta_fraction * width)
        v = alpha * v + current - threshold * z
        candidates = np.flatnonzero(v >= threshold)
        if inhibition == "kwta" and candidates.size > kwta:
            keep = candidates[np.argpartition(-v[candidates], kwta)[:kwta]]
            z_new = np.zeros(width, dtype=np.float32)
            z_new[keep] = 1.0
        else:
            z_new = (v >= threshold).astype(np.float32)
        z_new[refractory > 0] = 0.0
        z = z_new
        refractory = np.where(z > 0, 2, np.maximum(refractory - 1, 0))
        spike_count += z
        total_spikes += float(z.sum())
        adaptation = rho * adaptation + z

    if readout_mode == "spikes":
        state = spike_count
    elif readout_mode == "membrane":
        state = v
    else:
        state = spike_count + 0.5 * v / max(1e-6, float(np.abs(v).max()))
    return (
        np.argsort(-readout.matvec(state), kind="stable"),
        total_spikes / steps,
        float((spike_count > 0).mean()),
    )


def synthetic_codes(n: int, k: int = 64, width: int = 4096) -> SparseCodes:
    rows = np.arange(n, dtype=np.int64)[:, None]
    cols = (np.arange(k, dtype=np.int64)[None, :] * 7 + rows * 13) % width
    weights = ((np.arange(k, dtype=np.int64)[None, :] + rows) % 9 + 1) / 10.0
    return SparseCodes(idx=cols.astype(np.int32), weights=weights.astype(np.float32), width=width)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--lags", default="2,3,5")
    parser.add_argument(
        "--out", type=Path, default=Path("sweep_results/axo_spiking_sparse/scan.json")
    )
    args = parser.parse_args(argv)

    probe = base.load_probe_module()
    axo = probe.Axo(args.axo_src)
    items, temporary = base.load_items(args.data_root, axo)
    try:
        lex = [item.lex_sdr for item in items]
        df = probe.document_frequency(lex, axo.width)
        codes = probe.encode_arm("idf-2.0", lex, df, len(items), args.k, axo)
        sparse = sparse_from_codes(codes, axo.width)
        dense = dense_from_sparse(sparse)
        ids = [item.memory_id for item in items]
        index_of = {memory_id: index for index, memory_id in enumerate(ids)}

        # 静态：稀疏/稠密一致性 + 脉冲。
        probes = []
        targets = []
        for index, code in enumerate(codes):
            probes.append(base.dense_probe(axo.sdr.drop_bits(code, 0.5), axo.width))
            targets.append(index)
        sparse_orders = [rate_static_sparse(sparse, p) for p in probes]
        dense_orders = [rate_static_dense(dense, p) for p in probes]
        sparse_rate = base.top_metrics(sparse_orders, targets)
        dense_rate = base.top_metrics(dense_orders, targets)
        same_top1 = sum(int(a[0] == b[0]) for a, b in zip(sparse_orders, dense_orders, strict=True))
        print(
            f"sparse vs dense top1 agreement {same_top1}/{len(probes)}; "
            f"sparse {sparse_rate['top1']:.4f} dense {dense_rate['top1']:.4f}"
        )

        static_configs = {
            "membrane-none-h0-b0": {
                "readout_mode": "membrane",
                "inhibition": "none",
                "hetero": 0.0,
                "beta": 0.0,
            },
        }
        static_rows = []
        for name, config in static_configs.items():
            orders = []
            spikes = []
            active = []
            started = time.perf_counter()
            for p in probes:
                order, rate, act = simulate_sparse(
                    sparse,
                    sparse,
                    sparse,
                    p,
                    steps=10,
                    theta=0.5,
                    rho=0.9,
                    kwta_fraction=0.05,
                    global_gain=1.0,
                    **config,
                )
                orders.append(order)
                spikes.append(rate)
                active.append(act)
            elapsed = time.perf_counter() - started
            row = {
                "arm": "static-spiking",
                "config": name,
                **base.top_metrics(orders, targets),
                "spikes_per_step": float(np.mean(spikes)),
                "active": float(np.mean(active)),
                "elapsed_s": elapsed,
            }
            static_rows.append(row)
            print(
                f"static {name}: top1 {row['top1']:.4f} top5 {row['top5']:.4f} "
                f"spikes {row['spikes_per_step']:.1f} {elapsed:.2f}s"
            )

        # 序列遍历：W_seq 只用相邻轮 t -> t+1 学。
        turns: dict[tuple[str, int], int] = {}
        for item in items:
            if item.source_ref.startswith("turn://") and item.meta.get("turn_seq") is not None:
                turns[(item.session_id, int(item.meta["turn_seq"]))] = index_of[item.memory_id]
        prev_indices = []
        next_indices = []
        for (session, turn), index in sorted(turns.items()):
            following = turns.get((session, turn + 1))
            if following is not None:
                prev_indices.append(index)
                next_indices.append(following)
        prev_codes = sparse.subset(prev_indices)
        next_codes = sparse.subset(next_indices)

        configs = {
            "membrane-none-h0-b0": {
                "readout_mode": "membrane",
                "inhibition": "none",
                "hetero": 0.0,
                "beta": 0.0,
            },
            "membrane-none-h05-b0": {
                "readout_mode": "membrane",
                "inhibition": "none",
                "hetero": 0.5,
                "beta": 0.0,
            },
            "membrane-global-h05-b1": {
                "readout_mode": "membrane",
                "inhibition": "global",
                "hetero": 0.5,
                "beta": 1.0,
            },
        }
        traversal_rows = []
        for lag in [int(value) for value in args.lags.split(",") if value.strip()]:
            query_indices = []
            target_indices = []
            for (session, turn), index in sorted(turns.items()):
                target = turns.get((session, turn + lag))
                if target is not None:
                    query_indices.append(index)
                    target_indices.append(target)
            query_probes = [
                base.dense_probe(axo.sdr.drop_bits(codes[index], 0.5), axo.width)
                for index in query_indices
            ]
            one_orders = []
            iter_orders = []
            for p in query_probes:
                initial = base.normalize(next_codes.transpose_matvec(prev_codes.matvec(p)))
                one_orders.append(np.argsort(-sparse.matvec(initial), kind="stable"))
                state = initial
                for _ in range(lag - 1):
                    state = base.normalize(next_codes.transpose_matvec(prev_codes.matvec(state)))
                iter_orders.append(np.argsort(-sparse.matvec(state), kind="stable"))
            one = base.top_metrics(one_orders, target_indices)
            iterative = base.top_metrics(iter_orders, target_indices)
            print(
                f"lag {lag}: rate-one {one['top1']:.4f}/{one['top5']:.4f} "
                f"rate-iter {iterative['top1']:.4f}/{iterative['top5']:.4f} n={one['n']}"
            )
            for name, config in configs.items():
                orders = []
                spikes = []
                active = []
                started = time.perf_counter()
                for p in query_probes:
                    order, rate, act = simulate_sparse(
                        sparse,
                        prev_codes,
                        next_codes,
                        p,
                        steps=lag,
                        theta=0.5,
                        rho=0.9,
                        kwta_fraction=0.05,
                        global_gain=1.0,
                        **config,
                    )
                    orders.append(order)
                    spikes.append(rate)
                    active.append(act)
                elapsed = time.perf_counter() - started
                row = {
                    "lag": lag,
                    "arm": "spiking",
                    "config": name,
                    **base.top_metrics(orders, target_indices),
                    "spikes_per_step": float(np.mean(spikes)),
                    "active": float(np.mean(active)),
                    "elapsed_s": elapsed,
                }
                traversal_rows.append(row)
                print(
                    f"  {name}: top1 {row['top1']:.4f} top5 {row['top5']:.4f} "
                    f"spikes {row['spikes_per_step']:.1f} {elapsed:.2f}s"
                )

        # 规模：只算存储与单次查询；稠密只到 10,000。
        scale_rows = []
        for n in (200, 2000, 10000, 50000):
            codes_n = synthetic_codes(n)
            probe_vector = np.zeros(codes_n.width, dtype=np.float32)
            probe_vector[codes_n.idx[0]] = codes_n.weights[0]
            started = time.perf_counter()
            for _ in range(3):
                rate_static_sparse(codes_n, probe_vector)
            sparse_ms = (time.perf_counter() - started) / 3 * 1000
            dense_bytes = n * codes_n.width * 4
            dense_ms = None
            if n <= 10000:
                matrix = dense_from_sparse(codes_n)
                started = time.perf_counter()
                for _ in range(3):
                    rate_static_dense(matrix, probe_vector)
                dense_ms = (time.perf_counter() - started) / 3 * 1000
            scale_rows.append(
                {
                    "n": n,
                    "sparse_bytes": int(codes_n.idx.nbytes + codes_n.weights.nbytes),
                    "dense_bytes": int(dense_bytes),
                    "sparse_ms": sparse_ms,
                    "dense_ms": dense_ms,
                }
            )
            print(
                f"scale n={n}: sparse {sparse_ms:.2f} ms / "
                f"{codes_n.idx.nbytes + codes_n.weights.nbytes} B; "
                f"dense {dense_ms if dense_ms is None else round(dense_ms, 2)} ms / {dense_bytes} B"
            )

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "n_memories": len(items),
                    "sparse_rate": sparse_rate,
                    "dense_rate": dense_rate,
                    "sparse_dense_top1_agreement": same_top1 / len(probes),
                    "static": static_rows,
                    "traversal": traversal_rows,
                    "scale": scale_rows,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[落盘] {args.out}")
    finally:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
