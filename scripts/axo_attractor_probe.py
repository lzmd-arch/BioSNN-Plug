"""脉冲吸引子补全探针：LIF / ALIF 迭代 vs Axo 的一步 W。

预注册的基质编码结果见 [`sweep_results/axo_substrate_prereg.md`](../sweep_results/axo_substrate_prereg.md)。
本脚本只回答一件事：**在同一份关联码上，把一步 ``W p`` 换成 T 步 LIF / ALIF 动力学，
补全 top-1 会不会变好。**

用法::

    uv run python scripts/axo_attractor_probe.py --arm idf-2.0 --scale 200,500
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


def load_probe_module():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import axo_substrate_probe as probe

    return probe


def dense_codes(codes, width: int) -> np.ndarray:
    """``(n, width)`` 的加权指示矩阵——W = AᵀA 的隐式表示。"""
    matrix = np.zeros((len(codes), width), dtype=np.float32)
    for row, code in enumerate(codes):
        matrix[row, code.idx] = code.weights
    return matrix


def dense_probe(probe_sdr, width: int) -> np.ndarray:
    vector = np.zeros(width, dtype=np.float32)
    vector[probe_sdr.idx] = probe_sdr.weights
    return vector


def one_step_order(matrix: np.ndarray, probe: np.ndarray) -> np.ndarray:
    """Axo 现有的一步补全：``score = A·(Aᵀ(A·p))``。"""
    activation = matrix.T @ (matrix @ probe)
    scores = matrix @ activation
    return np.argsort(-scores, kind="stable")


def attractor_order(
    matrix: np.ndarray,
    probe: np.ndarray,
    *,
    steps: int,
    alpha: float,
    theta: float,
    beta: float,
    rho: float,
    input_gain: float,
    rec_gain: float,
    n_refractory: int,
    readout: str,
) -> tuple[np.ndarray, float, float]:
    """T 步 LIF / ALIF 吸引子。``beta=0`` 就是 LIF。"""
    width = matrix.shape[1]
    initial = matrix.T @ (matrix @ probe)
    scale = float(initial.max())
    if scale > 0:
        initial = initial / scale

    v = np.zeros(width, dtype=np.float32)
    z = np.zeros(width, dtype=np.float32)
    adaptation = np.zeros(width, dtype=np.float32)
    refractory = np.zeros(width, dtype=np.int32)
    spike_count = np.zeros(width, dtype=np.float32)
    total_spikes = 0.0

    for _ in range(steps):
        recurrent = matrix.T @ (matrix @ z)
        rec_scale = float(recurrent.max())
        if rec_scale > 0:
            recurrent = recurrent / rec_scale
        threshold = theta + beta * adaptation
        v = alpha * v + input_gain * initial + rec_gain * recurrent - threshold * z
        can_fire = (refractory == 0) & (v >= threshold)
        z = can_fire.astype(np.float32)
        refractory = np.where(can_fire, n_refractory, np.maximum(refractory - 1, 0))
        spike_count += z
        total_spikes += float(z.sum())
        adaptation = rho * adaptation + z

    scores = matrix @ (spike_count if readout == "spikes" else v)
    order = np.argsort(-scores, kind="stable")
    return order, total_spikes / steps, float((spike_count > 0).mean())


def evaluate_arm(
    matrix: np.ndarray,
    probes: list[np.ndarray],
    targets: list[str],
    ids: list[str],
    *,
    steps: int,
    alpha: float,
    theta: float,
    beta: float,
    rho: float,
    input_gain: float,
    rec_gain: float,
    n_refractory: int,
    readout: str,
) -> dict:
    top1 = 0
    top5 = 0
    spike_rates = []
    active = []
    for index, probe in enumerate(probes):
        order, spikes_per_step, active_fraction = attractor_order(
            matrix,
            probe,
            steps=steps,
            alpha=alpha,
            theta=theta,
            beta=beta,
            rho=rho,
            input_gain=input_gain,
            rec_gain=rec_gain,
            n_refractory=n_refractory,
            readout=readout,
        )
        ranked = [ids[int(position)] for position in order[:5]]
        top1 += ranked[0] == targets[index]
        top5 += targets[index] in ranked
        spike_rates.append(spikes_per_step)
        active.append(active_fraction)
    return {
        "top1": top1 / len(probes),
        "top5": top5 / len(probes),
        "spikes_per_step": float(np.mean(spike_rates)),
        "active_fraction": float(np.mean(active)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--corpus-dir", type=Path, default=AXO_CORPUS_DEFAULT)
    parser.add_argument("--arm", default="idf-2.0")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--scale", default="200,500")
    parser.add_argument("--steps", default="10")
    parser.add_argument("--theta", default="0.3,0.5,0.7")
    parser.add_argument("--beta", default="0.0,1.0")
    parser.add_argument("--rho", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--input-gain", type=float, default=1.0)
    parser.add_argument("--rec-gain", default="1.0")
    parser.add_argument("--n-refractory", type=int, default=1)
    parser.add_argument("--readout", default="spikes", choices=["spikes", "membrane"])
    parser.add_argument(
        "--out", type=Path, default=Path("sweep_results/axo_attractor_probe/scan.json")
    )
    args = parser.parse_args(argv)

    probe = load_probe_module()
    axo = probe.Axo(args.axo_src)
    base_corpus = axo.load(args.corpus_dir)

    rows = []
    for scale in [int(value) for value in args.scale.split(",") if value.strip()]:
        memories = axo.bench.scaled_memories(base_corpus.memories, scale)
        corpus = axo.bench.SubstrateCorpus(
            memories=tuple(memories), queries=base_corpus.queries, corpus_dir=base_corpus.corpus_dir
        )
        lex_codes = probe.lexical_codes(corpus, axo)
        df = probe.document_frequency(lex_codes, axo.width)
        codes = probe.encode_arm(args.arm, lex_codes, df, len(memories), args.k, axo)
        matrix = dense_codes(codes, axo.width)
        ids = [memory.memory_id for memory in memories]

        sampled = axo.bench.probe_sample(memories, 200)
        {memory.memory_id: memory for memory in memories}
        index_of = {memory.memory_id: index for index, memory in enumerate(memories)}
        probes = []
        probe_ids = []
        for memory in sampled:
            index = index_of[memory.memory_id]
            probes.append(dense_probe(axo.sdr.drop_bits(codes[index], 0.5), axo.width))
            probe_ids.append(memory.memory_id)

        # 一步基线：同一批探针、同一个矩阵。
        one_step_top1 = 0
        one_step_top5 = 0
        for index, probe_vector in enumerate(probes):
            order = one_step_order(matrix, probe_vector)
            ranked = [ids[int(position)] for position in order[:5]]
            one_step_top1 += ranked[0] == probe_ids[index]
            one_step_top5 += probe_ids[index] in ranked
        print(
            f"scale={scale} arm={args.arm} 一步 W：top1 {one_step_top1 / len(probes):.4f} "
            f"top5 {one_step_top5 / len(probes):.4f}"
        )

        for steps in [int(value) for value in args.steps.split(",") if value.strip()]:
            for beta in [float(value) for value in args.beta.split(",") if value.strip()]:
                for theta in [float(value) for value in args.theta.split(",") if value.strip()]:
                    for rec_gain in [
                        float(value) for value in args.rec_gain.split(",") if value.strip()
                    ]:
                        started = time.perf_counter()
                        result = evaluate_arm(
                            matrix,
                            probes,
                            probe_ids,
                            ids,
                            steps=steps,
                            alpha=args.alpha,
                            theta=theta,
                            beta=beta,
                            rho=args.rho,
                            input_gain=args.input_gain,
                            rec_gain=rec_gain,
                            n_refractory=args.n_refractory,
                            readout=args.readout,
                        )
                        elapsed = time.perf_counter() - started
                        row = {
                            "scale": scale,
                            "arm": args.arm,
                            "k": args.k,
                            "steps": steps,
                            "beta": beta,
                            "theta": theta,
                            "rec_gain": rec_gain,
                            "rho": args.rho,
                            "alpha": args.alpha,
                            "input_gain": args.input_gain,
                            "readout": args.readout,
                            "one_step_top1": one_step_top1 / len(probes),
                            "one_step_top5": one_step_top5 / len(probes),
                            "elapsed_s": elapsed,
                            **result,
                        }
                        rows.append(row)
                        print(
                            f"  steps={steps:<2} beta={beta:<4} theta={theta:<4} "
                            f"rg={rec_gain:<4} top1 {row['top1']:.4f} top5 {row['top5']:.4f} "
                            f"spikes/step {row['spikes_per_step']:.1f} "
                            f"active {row['active_fraction']:.2f} {elapsed:.1f}s"
                        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"records": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\n[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
