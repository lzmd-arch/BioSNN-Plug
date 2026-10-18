"""Axo 真脉冲原型：LIF/ALIF 逐步膜电位 + 离散脉冲。

预注册：[`sweep_results/axo_spiking_prototype_prereg.md`](../sweep_results/axo_spiking_prototype_prereg.md)。

用法::

    uv run python scripts/axo_spiking_prototype.py --data-root D:/AxoData

它只读真实库的副本；不写库、不输出正文。输出是每个配置一行 JSON 的聚合指标。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

AXO_SRC_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/src")
DATA_ROOT_DEFAULT = Path("D:/AxoData")


def load_probe_module():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import axo_substrate_probe as probe

    return probe


def copy_db(src: Path) -> Path:
    directory = Path(tempfile.mkdtemp(prefix="axo_spiking_"))
    destination = directory / "mem.db"
    source = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True)
    target = sqlite3.connect(destination)
    source.backup(target)
    target.close()
    source.close()
    return directory


def load_items(data_root: Path, axo):
    from memcore.store import MemoryStore

    directory = copy_db(data_root / "memory" / "mem.db")
    store = MemoryStore(directory / "mem.db")
    items = [
        item for item in store.load_all() if item.lex_sdr is not None and not item.is_persona_anchor
    ]
    store.close()
    return items, directory


def dense(codes, width: int) -> np.ndarray:
    matrix = np.zeros((len(codes), width), dtype=np.float32)
    for row, code in enumerate(codes):
        matrix[row, code.idx] = code.weights
    return matrix


def dense_probe(code, width: int) -> np.ndarray:
    vector = np.zeros(width, dtype=np.float32)
    vector[code.idx] = code.weights
    return vector


def normalize(values: np.ndarray) -> np.ndarray:
    peak = float(values.max()) if values.size else 0.0
    return values / peak if peak > 0.0 else np.zeros_like(values)


def static_rate_order(matrix: np.ndarray, probe: np.ndarray) -> np.ndarray:
    activation = matrix.T @ (matrix @ probe)
    return np.argsort(-(matrix @ activation), kind="stable")


def simulate(
    matrix: np.ndarray,
    probe: np.ndarray,
    *,
    prev: np.ndarray,
    nxt: np.ndarray,
    steps: int,
    theta: float,
    beta: float,
    rho: float,
    hetero: float,
    readout: str,
    inhibition: str,
    kwta_fraction: float,
    global_gain: float,
) -> tuple[np.ndarray, float, float]:
    """真的逐步 LIF/ALIF。``beta=0`` 是 LIF。返回 ``(order, spikes/step, active)``。"""
    width = matrix.shape[1]
    if prev.ndim == 1 and nxt.ndim == 1:
        initial = normalize(nxt * (prev @ probe))
    else:
        initial = normalize(nxt.T @ (prev @ probe))
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
        if prev.ndim == 1 and nxt.ndim == 1:
            recurrent = normalize(nxt * (prev @ z))
        else:
            recurrent = normalize(nxt.T @ (prev @ z))
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

    if readout == "spikes":
        state = spike_count
    elif readout == "membrane":
        state = v
    else:
        membrane = v / max(1e-6, float(np.abs(v).max()))
        state = spike_count + 0.5 * membrane
    order = np.argsort(-(matrix @ state), kind="stable")
    return order, total_spikes / steps, float((spike_count > 0).mean())


def top_metrics(orders: list[np.ndarray], targets: list[int]) -> dict[str, float]:
    top1 = top5 = 0
    for order, target in zip(orders, targets, strict=True):
        ranked = order[:5]
        top1 += int(ranked[0] == target)
        top5 += int(target in ranked)
    total = len(orders)
    return {"top1": top1 / total, "top5": top5 / total, "n": total}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--kwta-fraction", type=float, default=0.05)
    parser.add_argument("--sequence-drop", type=float, default=0.0)
    parser.add_argument("--sequence-lag", type=int, default=1)
    parser.add_argument("--readouts", default="spikes,membrane,hybrid")
    parser.add_argument("--inhibitions", default="none,global,kwta")
    parser.add_argument("--heteros", default="0,0.5")
    parser.add_argument("--betas", default="0,1")
    parser.add_argument(
        "--out", type=Path, default=Path("sweep_results/axo_spiking_prototype/scan.json")
    )
    args = parser.parse_args(argv)

    probe = load_probe_module()
    axo = probe.Axo(args.axo_src)
    items, temporary = load_items(args.data_root, axo)
    try:
        lex = [item.lex_sdr for item in items]
        df = probe.document_frequency(lex, axo.width)
        codes = probe.encode_arm("idf-2.0", lex, df, len(items), args.k, axo)
        matrix = dense(codes, axo.width)
        ids = [item.memory_id for item in items]
        index_of = {memory_id: index for index, memory_id in enumerate(ids)}

        # 静态探针：50% 丢位，找回自己。
        probes = []
        targets = []
        for index, code in enumerate(codes):
            probes.append(dense_probe(axo.sdr.drop_bits(code, 0.5), axo.width))
            targets.append(index)

        rate_orders = [static_rate_order(matrix, probe_vector) for probe_vector in probes]
        rate_static = top_metrics(rate_orders, targets)

        # 时序对：同一会话里相邻 turn。
        turns: dict[tuple[str, int], int] = {}
        for item in items:
            if item.source_ref.startswith("turn://") and item.meta.get("turn_seq") is not None:
                turns[(item.session_id, int(item.meta["turn_seq"]))] = index_of[item.memory_id]
        pairs = []
        for (session, turn), index in sorted(turns.items()):
            following = turns.get((session, turn + args.sequence_lag))
            if following is not None:
                pairs.append((index, following))

        seq_probes = [
            dense_probe(axo.sdr.drop_bits(codes[prev], args.sequence_drop), axo.width)
            for prev, _ in pairs
        ]
        seq_targets = [next_ for _, next_ in pairs]
        rate_seq_orders = []
        for (prev, next_), probe_vector in zip(pairs, seq_probes, strict=True):
            initial = normalize(matrix[next_] * (matrix[prev] @ probe_vector))
            rate_seq_orders.append(np.argsort(-(matrix @ initial), kind="stable"))
        rate_seq = top_metrics(rate_seq_orders, seq_targets)

        rows = [
            {"arm": "rate-static", **rate_static, "spikes_per_step": None, "active": None},
            {"arm": "rate-sequence", **rate_seq, "spikes_per_step": None, "active": None},
        ]
        readouts = [value.strip() for value in args.readouts.split(",") if value.strip()]
        inhibitions = [value.strip() for value in args.inhibitions.split(",") if value.strip()]
        heteros = [float(value) for value in args.heteros.split(",") if value.strip()]
        betas = [float(value) for value in args.betas.split(",") if value.strip()]
        for readout in readouts:
            for inhibition in inhibitions:
                for hetero in heteros:
                    for beta in betas:
                        started = time.perf_counter()
                        static_orders = []
                        static_spikes = []
                        static_active = []
                        for _index, probe_vector in enumerate(probes):
                            order, spikes, active = simulate(
                                matrix,
                                probe_vector,
                                prev=matrix,
                                nxt=matrix,
                                steps=args.steps,
                                theta=args.theta,
                                beta=beta,
                                rho=args.rho,
                                hetero=hetero,
                                readout=readout,
                                inhibition=inhibition,
                                kwta_fraction=args.kwta_fraction,
                                global_gain=1.0,
                            )
                            static_orders.append(order)
                            static_spikes.append(spikes)
                            static_active.append(active)
                        seq_orders = []
                        seq_spikes = []
                        seq_active = []
                        for (prev, next_), probe_vector in zip(pairs, seq_probes, strict=True):
                            order, spikes, active = simulate(
                                matrix,
                                probe_vector,
                                prev=matrix[prev],
                                nxt=matrix[next_],
                                steps=args.steps,
                                theta=args.theta,
                                beta=beta,
                                rho=args.rho,
                                hetero=hetero,
                                readout=readout,
                                inhibition=inhibition,
                                kwta_fraction=args.kwta_fraction,
                                global_gain=1.0,
                            )
                            seq_orders.append(order)
                            seq_spikes.append(spikes)
                            seq_active.append(active)
                        elapsed = time.perf_counter() - started
                        row = {
                            "arm": "spiking",
                            "readout": readout,
                            "inhibition": inhibition,
                            "hetero": hetero,
                            "beta": beta,
                            "static": top_metrics(static_orders, targets),
                            "sequence": top_metrics(seq_orders, seq_targets),
                            "spikes_per_step": float(np.mean(static_spikes + seq_spikes)),
                            "active": float(np.mean(static_active + seq_active)),
                            "elapsed_s": elapsed,
                        }
                        rows.append(row)
                        print(
                            f"{readout:<8} {inhibition:<6} hetero={hetero:<3} beta={beta:<3} "
                            f"static {row['static']['top1']:.4f}/{row['static']['top5']:.4f}  "
                            f"sequence {row['sequence']['top1']:.4f}/{row['sequence']['top5']:.4f}  "
                            f"spikes {row['spikes_per_step']:.1f}  {elapsed:.1f}s"
                        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "n_memories": len(items),
                    "n_pairs": len(pairs),
                    "rate_static": rate_static,
                    "rate_sequence": rate_seq,
                    "records": rows,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nrate static {rate_static} / sequence {rate_seq}")
        print(f"[落盘] {args.out}")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
