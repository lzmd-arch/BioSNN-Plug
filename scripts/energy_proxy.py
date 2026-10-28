"""阶段四能效的代理计算：脉冲操作（SOP） vs ANN MAC，不是硬件功率。

**这是代理，不是测量**。没有功率计，也没有逐样本能耗采集；本脚本只把
“脉冲稀疏度 × 突触操作数”和“等效 ANN 的 MAC 数”放在同一张表上，使用文献常见的
45nm 能量常数（**未在本仓库核验**），并明确标注不能与硬件能耗混报。
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "sweep_results" / "phase4_energy" / "summary.json"

#: 文献常见的 45nm 量级（Horowitz, 2014）；本仓库未做硬件核验。
ENERGY_PER_MAC_PJ = 4.6
ENERGY_PER_SOP_PJ = 0.9


def proxy(n_in: int, n_rec: int, steps: int, sparsity: float) -> dict[str, float]:
    input_synapses = n_in * n_rec
    recurrent_synapses = n_rec * n_rec
    sop_per_step = (input_synapses + recurrent_synapses) * (1.0 - sparsity)
    mac_per_step = input_synapses + recurrent_synapses
    sop = steps * sop_per_step
    mac = steps * mac_per_step
    snn_energy = sop * ENERGY_PER_SOP_PJ
    ann_energy = mac * ENERGY_PER_MAC_PJ
    return {
        "n_in": n_in,
        "n_rec": n_rec,
        "steps": steps,
        "sparsity": sparsity,
        "sop_per_sample": sop,
        "mac_per_sample": mac,
        "snn_energy_pj": snn_energy,
        "ann_energy_pj": ann_energy,
        "ann_over_snn": ann_energy / snn_energy if snn_energy else float("inf"),
    }


def main() -> int:
    configs = [
        proxy(28, 256, 28, 0.8758183717727661),
        proxy(28, 256, 28, 0.9),
        proxy(28, 256, 28, 0.95),
        proxy(28, 256, 28, 0.98),
    ]
    summary = {
        "kind": "analytic proxy, NOT hardware energy",
        "energy_constants_pj": {
            "MAC": ENERGY_PER_MAC_PJ,
            "SOP": ENERGY_PER_SOP_PJ,
            "source": "45nm order-of-magnitude values; unverified in this repo",
        },
        "assumptions": "inference only; dense ANN with the same layer shape; sparsity is a parameter",
        "configs": configs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    for config in configs:
        print(f"sparsity {config['sparsity']:.2f}: ANN/SNN proxy {config['ann_over_snn']:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
