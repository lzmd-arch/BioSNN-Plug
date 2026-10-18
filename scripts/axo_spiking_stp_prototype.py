"""用短时突触抑制（STP）表达顺序的原型。

预注册：[`sweep_results/axo_spiking_stp_prereg.md`](../sweep_results/axo_spiking_stp_prereg.md)。

同一会话相邻轮的 A、B 分两步给；目标都是先到的那个。STP 让第一个输入用掉共享位的
突触资源，第二个输入被削弱。真库只读副本；不输出正文。
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


def simulate_stp(
    readout: sparse_mod.SparseCodes,
    codes: sparse_mod.SparseCodes,
    inputs: list[np.ndarray],
    *,
    alpha: float,
    theta: float,
    depression: float,
    recovery: float,
) -> tuple[np.ndarray, float, float]:
    currents = [codes.transpose_matvec(codes.matvec(item)) for item in inputs]
    scale = max((float(current.max()) for current in currents), default=0.0)
    if scale > 0:
        currents = [current / scale for current in currents]

    width = readout.width
    efficacy = np.ones(width, dtype=np.float32)
    v = np.zeros(width, dtype=np.float32)
    z = np.zeros(width, dtype=np.float32)
    refractory = np.zeros(width, dtype=np.int32)
    spike_count = np.zeros(width, dtype=np.float32)
    total_spikes = 0.0

    for current, raw in zip(currents, inputs, strict=True):
        recurrent = base.normalize(codes.transpose_matvec(codes.matvec(z)))
        injected = current * efficacy + recurrent
        v = alpha * v + injected - theta * z
        z = (v >= theta).astype(np.float32)
        z[refractory > 0] = 0.0
        refractory = np.where(z > 0, 2, np.maximum(refractory - 1, 0))
        spike_count += z
        total_spikes += float(z.sum())
        active = (raw > 0).astype(np.float32)
        efficacy = np.clip(efficacy - depression * active + recovery * (1.0 - efficacy), 0.0, 1.0)

    return (
        np.argsort(-readout.matvec(v), kind="stable"),
        total_spikes / max(1, len(inputs)),
        float((spike_count > 0).mean()),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path("sweep_results/axo_spiking_stp/scan.json"))
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
        pairs = []
        for (session, turn), index in sorted(turns.items()):
            following = turns.get((session, turn + 1))
            if following is not None:
                pairs.append((index, following))
        zero = np.zeros(axo.width, dtype=np.float32)

        def inputs_for(first: int, second: int) -> list[np.ndarray]:
            return [
                base.dense_probe(codes[first], axo.width),
                base.dense_probe(codes[second], axo.width),
                zero,
                zero,
                zero,
            ]

        arms = {
            "lif-a095": {"alpha": 0.95, "depression": 0.0, "recovery": 0.0},
            "lif-a098": {"alpha": 0.98, "depression": 0.0, "recovery": 0.0},
            "stp-d02": {"alpha": 0.98, "depression": 0.2, "recovery": 0.05},
            "stp-d05": {"alpha": 0.98, "depression": 0.5, "recovery": 0.05},
            "stp-d08": {"alpha": 0.98, "depression": 0.8, "recovery": 0.05},
        }
        rows = []
        for name, config in arms.items():
            forward = []
            backward = []
            spikes = []
            for first, second in pairs:
                for inputs, target, bucket in (
                    (inputs_for(first, second), first, forward),
                    (inputs_for(second, first), second, backward),
                ):
                    order, rate, _active = simulate_stp(
                        sparse,
                        sparse,
                        inputs,
                        alpha=float(config["alpha"]),
                        theta=0.5,
                        depression=float(config["depression"]),
                        recovery=float(config["recovery"]),
                    )
                    bucket.append(int(order[0]) == target)
                    spikes.append(rate)
            forward_rate = float(np.mean(forward))
            backward_rate = float(np.mean(backward))
            average = (forward_rate + backward_rate) / 2
            row = {
                "arm": name,
                "alpha": config["alpha"],
                "depression": config["depression"],
                "recovery": config["recovery"],
                "forward": forward_rate,
                "backward": backward_rate,
                "top1": average,
                "spikes_per_step": float(np.mean(spikes)),
            }
            rows.append(row)
            print(
                f"{name:<10} A→B {forward_rate:.4f} B→A {backward_rate:.4f} "
                f"avg {average:.4f} spikes {np.mean(spikes):.1f}"
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
