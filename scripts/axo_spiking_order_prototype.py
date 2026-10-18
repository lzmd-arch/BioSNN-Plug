"""时序顺序判别原型：A→B vs B→A，目标都是"先到的那个"。

预注册：[`sweep_results/axo_spiking_order_prereg.md`](../sweep_results/axo_spiking_order_prereg.md)。

线性泄漏积分器对顺序可交换，平均 top-1 的结构上限是 0.50；脉冲的适应/阈值可能
让先到的输入占优。真库只读副本；不输出正文。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import axo_spiking_noise_stream_prototype as noise_mod
import axo_spiking_prototype as base
import axo_spiking_sparse_prototype as sparse_mod

AXO_SRC_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/src")
DATA_ROOT_DEFAULT = Path("D:/AxoData")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--offset", type=int, default=37)
    parser.add_argument(
        "--out", type=Path, default=Path("sweep_results/axo_spiking_order/scan.json")
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
        count = len(codes)
        pairs = [(index, (index + args.offset) % count) for index in range(count)]
        zero = np.zeros(axo.width, dtype=np.float32)

        def inputs_for(first: int, second: int) -> list[np.ndarray]:
            return [
                base.dense_probe(codes[first], axo.width),
                base.dense_probe(codes[second], axo.width),
                zero,
                zero,
                zero,
            ]

        rate_first = []
        rate_last = []
        rate_integ = []
        for first, second in pairs:
            rate_first.append(
                int(
                    sparse_mod.rate_static_sparse(
                        sparse, base.dense_probe(codes[first], axo.width)
                    )[0]
                )
                == first
            )
            rate_last.append(
                int(
                    sparse_mod.rate_static_sparse(
                        sparse, base.dense_probe(codes[second], axo.width)
                    )[0]
                )
                == first
            )
            integrated = noise_mod.linear_integrate(
                sparse,
                sparse,
                sparse,
                inputs_for(first, second),
                alpha=0.95,
                leaky=True,
            )
            rate_integ.append(int(integrated[0]) == first)

        rows = [
            {
                "arm": "rate-first",
                "top1": float(np.mean(rate_first)),
            },
            {
                "arm": "rate-last",
                "top1": float(np.mean(rate_last)),
            },
            {
                "arm": "rate-integ",
                "top1": float(np.mean(rate_integ)),
            },
        ]
        print(
            f"rate-first {rows[0]['top1']:.4f} rate-last {rows[1]['top1']:.4f} "
            f"rate-integ {rows[2]['top1']:.4f}"
        )

        configs = {
            "lif-membrane": {
                "beta": 0.0,
                "rho": 0.9,
                "hetero": 0.0,
                "inhibition": "none",
            },
            "alif-b1-r05": {
                "beta": 1.0,
                "rho": 0.5,
                "hetero": 0.0,
                "inhibition": "none",
            },
            "alif-b1-r09": {
                "beta": 1.0,
                "rho": 0.9,
                "hetero": 0.0,
                "inhibition": "none",
            },
            "alif-b2-r09": {
                "beta": 2.0,
                "rho": 0.9,
                "hetero": 0.0,
                "inhibition": "none",
            },
            "alif-b1-r09-global": {
                "beta": 1.0,
                "rho": 0.9,
                "hetero": 0.0,
                "inhibition": "global",
            },
            "alif-b1-r09-global-h05": {
                "beta": 1.0,
                "rho": 0.9,
                "hetero": 0.5,
                "inhibition": "global",
            },
        }
        for name, config in configs.items():
            forward = []
            backward = []
            spike_rates = []
            for first, second in pairs:
                for inputs, target, bucket in (
                    (inputs_for(first, second), first, forward),
                    (inputs_for(second, first), second, backward),
                ):
                    order, spikes, _active = noise_mod.simulate_stream(
                        sparse,
                        sparse,
                        sparse,
                        inputs,
                        theta=0.5,
                        beta=float(config["beta"]),
                        rho=float(config["rho"]),
                        hetero=float(config["hetero"]),
                        readout_mode="membrane",
                        inhibition=str(config["inhibition"]),
                        kwta_fraction=0.05,
                        global_gain=1.0,
                    )
                    bucket.append(int(order[0]) == target)
                    spike_rates.append(spikes)
            forward_rate = float(np.mean(forward))
            backward_rate = float(np.mean(backward))
            average = (forward_rate + backward_rate) / 2
            rows.append(
                {
                    "arm": "spiking",
                    "config": name,
                    "forward": forward_rate,
                    "backward": backward_rate,
                    "top1": average,
                    "spikes_per_step": float(np.mean(spike_rates)),
                }
            )
            print(
                f"{name:<24} A→B {forward_rate:.4f} B→A {backward_rate:.4f} "
                f"avg {average:.4f} spikes {np.mean(spike_rates):.1f}"
            )

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"n": count, "records": rows}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[落盘] {args.out}")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
