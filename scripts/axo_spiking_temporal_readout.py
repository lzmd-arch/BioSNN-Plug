"""脉冲时序读出：用首次发放/越阈时刻给记忆打分。

预注册：[`sweep_results/axo_spiking_temporal_prereg.md`](../sweep_results/axo_spiking_temporal_prereg.md)。

顺序任务 A→B / B→A，目标都是先到的那个；读出改为
``score_m = Σ_i 1/(1+t_first(i))``。线性积分器用首次越阈时刻作同口径对照。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import axo_spiking_prototype as base
import axo_spiking_sparse_prototype as sparse_mod

AXO_SRC_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/src")
DATA_ROOT_DEFAULT = Path("D:/AxoData")


def first_times(
    codes: sparse_mod.SparseCodes,
    inputs: list[np.ndarray],
    *,
    alpha: float,
    theta: float,
    spiking: bool,
    beta: float,
    rho: float,
    hetero: float,
    inhibition: str,
) -> tuple[np.ndarray, np.ndarray]:
    currents = [codes.transpose_matvec(codes.matvec(item)) for item in inputs]
    scale = max((float(current.max()) for current in currents), default=0.0)
    if scale > 0:
        currents = [current / scale for current in currents]
    width = codes.width
    if hetero > 0.0:
        tau = 20.0 * np.exp(hetero * np.sin(np.arange(width, dtype=np.float64) * 0.61803398875))
        alpha_vec = np.exp(-1.0 / tau).astype(np.float32)
    else:
        alpha_vec = np.float32(alpha)

    v = np.zeros(width, dtype=np.float32)
    z = np.zeros(width, dtype=np.float32)
    adaptation = np.zeros(width, dtype=np.float32)
    refractory = np.zeros(width, dtype=np.int32)
    first_cross = np.full(width, np.inf, dtype=np.float64)
    first_spike = np.full(width, np.inf, dtype=np.float64)

    for step, current in enumerate(currents, start=1):
        recurrent = base.normalize(codes.transpose_matvec(codes.matvec(z))) if spiking else 0.0
        threshold = theta + beta * adaptation
        injected = current + recurrent
        if inhibition == "global" and spiking:
            injected = injected - float(z.sum()) / max(1.0, 0.05 * width)
        v = alpha_vec * v + injected - (threshold * z if spiking else 0.0)
        crossed = (v >= threshold) & np.isinf(first_cross)
        first_cross[crossed] = step
        if spiking:
            z_new = (v >= threshold).astype(np.float32)
            z_new[refractory > 0] = 0.0
            fresh = (z_new > 0) & np.isinf(first_spike)
            first_spike[fresh] = step
            z = z_new
            refractory = np.where(z > 0, 2, np.maximum(refractory - 1, 0))
            adaptation = rho * adaptation + z
    return first_cross, first_spike


def temporal_scores(codes: sparse_mod.SparseCodes, times: np.ndarray) -> np.ndarray:
    values = np.where(np.isfinite(times[codes.idx]), 1.0 / (1.0 + times[codes.idx]), 0.0)
    return (values * codes.weights).sum(axis=1)


def evaluate(
    code_sdrs,
    codes: sparse_mod.SparseCodes,
    pairs: list[tuple[int, int]],
    *,
    alpha: float,
    theta: float = 0.5,
    spiking: bool,
    beta: float = 0.0,
    rho: float = 0.9,
    hetero: float = 0.0,
    inhibition: str = "none",
) -> dict[str, float]:
    zeros = [np.zeros(codes.width, dtype=np.float32) for _ in range(3)]
    forward: list[int] = []
    backward: list[int] = []
    first_means = []
    for first, second in pairs:
        for order, target, bucket in (
            ([first, second], first, forward),
            ([second, first], second, backward),
        ):
            inputs = [
                base.dense_probe(code_sdrs[order[0]], codes.width),
                base.dense_probe(code_sdrs[order[1]], codes.width),
                *zeros,
            ]
            first_cross, first_spike = first_times(
                codes,
                inputs,
                alpha=alpha,
                theta=theta,
                spiking=spiking,
                beta=beta,
                rho=rho,
                hetero=hetero,
                inhibition=inhibition,
            )
            times = first_spike if spiking else first_cross
            scores = temporal_scores(codes, times)
            bucket.append(int(np.argmax(scores)) == target)
            active = times[np.isfinite(times)]
            first_means.append(float(active.mean()) if active.size else float("nan"))
    forward_rate = float(np.mean(forward))
    backward_rate = float(np.mean(backward))
    return {
        "forward": forward_rate,
        "backward": backward_rate,
        "top1": (forward_rate + backward_rate) / 2,
        "first_mean": float(np.nanmean(first_means)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--pair-mode", default="turns", choices=["turns", "offset"])
    parser.add_argument("--offset", type=int, default=37)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument(
        "--out", type=Path, default=Path("sweep_results/axo_spiking_temporal/scan.json")
    )
    args = parser.parse_args(argv)

    probe = base.load_probe_module()
    axo = probe.Axo(args.axo_src)
    items, temporary = base.load_items(args.data_root, axo)
    try:
        lex = [item.lex_sdr for item in items]
        df = probe.document_frequency(lex, axo.width)
        codes = probe.encode_arm("idf-2.0", lex, df, len(items), args.k, axo)
        sparse = sparse_mod.sparse_from_codes(codes, axo.width)
        index_of = {item.memory_id: index for index, item in enumerate(items)}
        turns: dict[tuple[str, int], int] = {}
        for item in items:
            if item.source_ref.startswith("turn://") and item.meta.get("turn_seq") is not None:
                turns[(item.session_id, int(item.meta["turn_seq"]))] = index_of[item.memory_id]
        if args.pair_mode == "turns":
            pairs = []
            for (session, turn), index in sorted(turns.items()):
                following = turns.get((session, turn + 1))
                if following is not None:
                    pairs.append((index, following))
        else:
            pairs = [(index, (index + args.offset) % len(codes)) for index in range(len(codes))]

        rows = [
            {
                "arm": "linear-a095",
                **evaluate(codes, sparse, pairs, alpha=0.95, theta=args.theta, spiking=False),
            },
            {
                "arm": "linear-a098",
                **evaluate(codes, sparse, pairs, alpha=0.98, theta=args.theta, spiking=False),
            },
            {
                "arm": "lif-a095",
                **evaluate(codes, sparse, pairs, alpha=0.95, theta=args.theta, spiking=True),
            },
            {
                "arm": "lif-a098",
                **evaluate(codes, sparse, pairs, alpha=0.98, theta=args.theta, spiking=True),
            },
            {
                "arm": "alif-b1-r09",
                **evaluate(
                    codes,
                    sparse,
                    pairs,
                    alpha=0.98,
                    theta=args.theta,
                    spiking=True,
                    beta=1.0,
                    rho=0.9,
                ),
            },
        ]
        for row in rows:
            print(
                f"{row['arm']:<12} A→B {row['forward']:.4f} B→A {row['backward']:.4f} "
                f"avg {row['top1']:.4f} first {row['first_mean']:.2f}"
            )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"n_pairs": len(pairs), "records": rows}, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        print(f"[落盘] {args.out}")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
