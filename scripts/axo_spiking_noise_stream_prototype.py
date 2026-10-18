"""真脉冲第三轮：噪声下的重复呈现、流式分片整合、候选子集延迟。

预注册：[`sweep_results/axo_spiking_noise_prereg.md`](../sweep_results/axo_spiking_noise_prereg.md)。

- 任务 A：同一探针重复 T 次，每次带不同的确定性位噪声；脉冲 vs 线性泄漏积分器。
- 任务 B：把探针活跃位切成 T 片逐步给；脉冲 vs 线性积分器 / 完整探针 / 第一片。
- 任务 C：50,000 条合成稀疏码的倒排候选子集，单次查询延迟。

真库只读副本；不输出正文。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import axo_spiking_prototype as base
import axo_spiking_sparse_prototype as sparse_mod

AXO_SRC_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/src")
DATA_ROOT_DEFAULT = Path("D:/AxoData")


def simulate_stream(
    readout: sparse_mod.SparseCodes,
    prev: sparse_mod.SparseCodes,
    nxt: sparse_mod.SparseCodes,
    inputs: list[np.ndarray],
    *,
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
    currents = [nxt.transpose_matvec(prev.matvec(item)) for item in inputs]
    scale = max((float(current.max()) for current in currents), default=0.0)
    if scale > 0:
        currents = [current / scale for current in currents]

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

    for current in currents:
        recurrent = base.normalize(nxt.transpose_matvec(prev.matvec(z)))
        threshold = theta + beta * adaptation
        injected = current + recurrent
        if inhibition == "global" and global_gain > 0.0:
            injected = injected - global_gain * float(z.sum()) / max(1.0, kwta_fraction * width)
        v = alpha * v + injected - threshold * z
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
        total_spikes / max(1, len(currents)),
        float((spike_count > 0).mean()),
    )


def linear_integrate(
    readout: sparse_mod.SparseCodes,
    prev: sparse_mod.SparseCodes,
    nxt: sparse_mod.SparseCodes,
    inputs: list[np.ndarray],
    *,
    alpha: float,
    leaky: bool,
) -> np.ndarray:
    currents = [nxt.transpose_matvec(prev.matvec(item)) for item in inputs]
    scale = max((float(current.max()) for current in currents), default=0.0)
    if scale > 0:
        currents = [current / scale for current in currents]
    v = np.zeros(readout.width, dtype=np.float64)
    for current in currents:
        v = (alpha * v if leaky else v) + current
    return np.argsort(-readout.matvec(v), kind="stable")


def noisy_probe(code, *, drop: float, noise: float, repetition: int, width: int) -> np.ndarray:
    base_vector = base.dense_probe(code, width)
    count = int(noise * max(1, code.k))
    if count <= 0:
        return base_vector
    existing = {int(bit) for bit in code.idx}
    positions = []
    offset = 0
    while len(positions) < count:
        candidate = (repetition * 7919 + offset * 104729) % width
        offset += 1
        if candidate not in existing and candidate not in positions:
            positions.append(candidate)
    vector = base_vector.copy()
    for position in positions:
        vector[position] = 0.5
    return vector


def split_probe(probe: np.ndarray, pieces: int) -> list[np.ndarray]:
    active = np.flatnonzero(probe)
    chunks: list[np.ndarray] = []
    for index in range(pieces):
        piece = np.zeros_like(probe)
        selected = active[index::pieces]
        piece[selected] = probe[selected]
        chunks.append(piece)
    return chunks


def build_inverted(codes: sparse_mod.SparseCodes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat_bits = codes.idx.ravel().astype(np.int64)
    rows = np.repeat(np.arange(codes.n, dtype=np.int64), codes.idx.shape[1])
    order = np.argsort(flat_bits, kind="stable")
    sorted_bits = flat_bits[order]
    sorted_rows = rows[order]
    counts = np.bincount(flat_bits, minlength=codes.width)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return sorted_bits, sorted_rows, offsets


def inverted_candidates(
    probe: np.ndarray, sorted_bits, sorted_rows, offsets, *, limit: int
) -> np.ndarray:
    bits = np.flatnonzero(probe)
    gathered = []
    for bit in bits:
        gathered.append(sorted_rows[offsets[bit] : offsets[bit + 1]])
    if not gathered:
        return np.zeros(0, dtype=np.int64)
    merged = np.concatenate(gathered)
    counts = np.bincount(merged, minlength=int(sorted_rows.max()) + 1)
    return np.argsort(-counts, kind="stable")[:limit]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--noises", default="0,0.05,0.10,0.20")
    parser.add_argument(
        "--out", type=Path, default=Path("sweep_results/axo_spiking_noise/scan.json")
    )
    args = parser.parse_args(argv)

    probe_module = base.load_probe_module()
    axo = probe_module.Axo(args.axo_src)
    items, temporary = base.load_items(args.data_root, axo)
    try:
        lex = [item.lex_sdr for item in items]
        df = probe_module.document_frequency(lex, axo.width)
        codes = probe_module.encode_arm("idf-2.0", lex, df, len(items), args.k, axo)
        sparse = sparse_mod.sparse_from_codes(codes, axo.width)
        targets = list(range(len(codes)))

        rows = []
        for noise in [float(value) for value in args.noises.split(",") if value.strip()]:
            probes = [
                noisy_probe(
                    code,
                    drop=0.5,
                    noise=noise,
                    repetition=rep,
                    width=axo.width,
                )
                for rep, code in enumerate(codes)
            ]
            # 每个目标重复 T 次，每次换一套确定性噪声。
            repetitions = [
                [
                    noisy_probe(code, drop=0.5, noise=noise, repetition=rep, width=axo.width)
                    for rep in range(args.steps)
                ]
                for code in codes
            ]
            rate_once = base.top_metrics(
                [sparse_mod.rate_static_sparse(sparse, item) for item in probes], targets
            )
            rate_integ = base.top_metrics(
                [
                    linear_integrate(sparse, sparse, sparse, item, alpha=0.95, leaky=True)
                    for item in repetitions
                ],
                targets,
            )
            spiking_lif = base.top_metrics(
                [
                    simulate_stream(
                        sparse,
                        sparse,
                        sparse,
                        item,
                        theta=0.5,
                        beta=0.0,
                        rho=0.9,
                        hetero=0.0,
                        readout_mode="membrane",
                        inhibition="none",
                        kwta_fraction=0.05,
                        global_gain=1.0,
                    )[0]
                    for item in repetitions
                ],
                targets,
            )
            spiking_alif = base.top_metrics(
                [
                    simulate_stream(
                        sparse,
                        sparse,
                        sparse,
                        item,
                        theta=0.5,
                        beta=1.0,
                        rho=0.9,
                        hetero=0.5,
                        readout_mode="membrane",
                        inhibition="global",
                        kwta_fraction=0.05,
                        global_gain=1.0,
                    )[0]
                    for item in repetitions
                ],
                targets,
            )
            row = {
                "task": "noise",
                "noise": noise,
                "rate_once": rate_once,
                "rate_integ": rate_integ,
                "spiking": spiking_lif,
                "spiking_alif": spiking_alif,
            }
            rows.append(row)
            print(
                f"noise {noise:.2f}: rate-once {rate_once['top1']:.4f} "
                f"rate-integ {rate_integ['top1']:.4f} "
                f"spiking {spiking_lif['top1']:.4f} / alif {spiking_alif['top1']:.4f}"
            )

        # 任务 B：流式分片。
        stream_probes = [
            base.dense_probe(axo.sdr.drop_bits(code, 0.5), axo.width) for code in codes
        ]
        chunks = [split_probe(item, args.steps) for item in stream_probes]
        rate_full = base.top_metrics(
            [sparse_mod.rate_static_sparse(sparse, item) for item in stream_probes], targets
        )
        rate_first = base.top_metrics(
            [sparse_mod.rate_static_sparse(sparse, item[0]) for item in chunks], targets
        )
        rate_integ = base.top_metrics(
            [
                linear_integrate(sparse, sparse, sparse, item, alpha=0.95, leaky=True)
                for item in chunks
            ],
            targets,
        )
        spiking_lif = base.top_metrics(
            [
                simulate_stream(
                    sparse,
                    sparse,
                    sparse,
                    item,
                    theta=0.5,
                    beta=0.0,
                    rho=0.9,
                    hetero=0.0,
                    readout_mode="membrane",
                    inhibition="none",
                    kwta_fraction=0.05,
                    global_gain=1.0,
                )[0]
                for item in chunks
            ],
            targets,
        )
        spiking_alif = base.top_metrics(
            [
                simulate_stream(
                    sparse,
                    sparse,
                    sparse,
                    item,
                    theta=0.5,
                    beta=1.0,
                    rho=0.9,
                    hetero=0.5,
                    readout_mode="membrane",
                    inhibition="global",
                    kwta_fraction=0.05,
                    global_gain=1.0,
                )[0]
                for item in chunks
            ],
            targets,
        )
        stream_row = {
            "task": "stream",
            "rate_full": rate_full,
            "rate_first": rate_first,
            "rate_integ": rate_integ,
            "spiking": spiking_lif,
            "spiking_alif": spiking_alif,
        }
        rows.append(stream_row)
        print(
            f"stream: rate-full {rate_full['top1']:.4f} rate-first {rate_first['top1']:.4f} "
            f"rate-integ {rate_integ['top1']:.4f} "
            f"spiking {spiking_lif['top1']:.4f} / alif {spiking_alif['top1']:.4f}"
        )

        # 任务 C：50,000 条合成码的倒排候选。
        synthetic = sparse_mod.synthetic_codes(50000)
        started = time.perf_counter()
        sorted_bits, sorted_rows, offsets = build_inverted(synthetic)
        build_s = time.perf_counter() - started
        probe_vector = np.zeros(synthetic.width, dtype=np.float32)
        probe_vector[synthetic.idx[0]] = synthetic.weights[0]
        full_order = sparse_mod.rate_static_sparse(synthetic, probe_vector)
        started = time.perf_counter()
        candidates = inverted_candidates(
            probe_vector, sorted_bits, sorted_rows, offsets, limit=2000
        )
        subset = synthetic.subset(candidates)
        activation = subset.transpose_matvec(subset.matvec(probe_vector))
        sub_order = np.argsort(-subset.matvec(activation), kind="stable")
        query_s = time.perf_counter() - started
        best = int(candidates[int(sub_order[0])])
        scale_row = {
            "task": "scale",
            "n": synthetic.n,
            "build_s": build_s,
            "query_s": query_s,
            "candidates": int(candidates.size),
            "top1_matches_full": bool(best == int(full_order[0])),
            "sparse_bytes": int(synthetic.idx.nbytes + synthetic.weights.nbytes),
            "dense_bytes": int(synthetic.n * synthetic.width * 4),
        }
        rows.append(scale_row)
        print(
            f"scale: build {build_s:.2f}s query {query_s * 1000:.2f}ms "
            f"candidates {candidates.size} top1_match {scale_row['top1_matches_full']}"
        )

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"records": rows}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[落盘] {args.out}")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
