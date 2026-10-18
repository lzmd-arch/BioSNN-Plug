"""汇总小参数模型的 LIF / ALIF 对照，并按预注册判据给出结论。

预注册：[`sweep_results/neuron_models_pilot_prereg.md`](../sweep_results/neuron_models_pilot_prereg.md)。

用法::

    uv run python scripts/analyze_neuron_models_pilot.py

读取 ``sweep_results/neuron_models_pilot/{lif,alif}/seed*.json``，打印两臂的均值 / 极差 /
次指标，并按 1.00 pp 的判据机械地判胜负。**不引入任何「看起来差不多」的判断**。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

#: 预注册里写死的胜差与稳定性闸门（绝对百分点）。
WIN_MARGIN_PP = 1.00
STABILITY_RANGE_PP = 5.00
RANDOM_BASELINE = 0.10


def load_arm(directory: Path) -> list[dict]:
    """读一个臂下所有种子 JSON，按种子号排序。"""
    records = []
    for path in sorted(directory.glob("seed*.json")):
        with path.open(encoding="utf-8") as handle:
            records.append(json.load(handle))
    if not records:
        raise SystemExit(f"{directory} 下没有 seed*.json。")
    return sorted(records, key=lambda record: int(record["seed"]))


def summarize(records: list[dict], label: str) -> dict:
    """算一个臂的均值与极差。"""
    accuracy = [float(record["test_accuracy"]) for record in records]
    active = [float(record["active_fraction"]) for record in records]
    peak_mb = [float(record["peak_mb"]) for record in records]
    elapsed = [float(record["elapsed_s"]) for record in records]
    return {
        "label": label,
        "n": len(records),
        "accuracy_mean": statistics.fmean(accuracy),
        "accuracy_min": min(accuracy),
        "accuracy_max": max(accuracy),
        "accuracy_range_pp": (max(accuracy) - min(accuracy)) * 100.0,
        "active_mean": statistics.fmean(active),
        "peak_mb_mean": statistics.fmean(peak_mb),
        "elapsed_mean_s": statistics.fmean(elapsed),
        "accuracy": accuracy,
        "active": active,
    }


def print_arm(summary: dict) -> None:
    print(
        f"{summary['label']:<5} n={summary['n']}  "
        f"准确率 均值 {summary['accuracy_mean']:.4f}  "
        f"区间 [{summary['accuracy_min']:.4f}, {summary['accuracy_max']:.4f}]  "
        f"极差 {summary['accuracy_range_pp']:.2f} pp  "
        f"活跃比例 {summary['active_mean']:.4f}  "
        f"显存 {summary['peak_mb_mean']:.1f} MiB  "
        f"耗时 {summary['elapsed_mean_s']:.1f} s"
    )
    print(f"      逐种子准确率：{[f'{value:.4f}' for value in summary['accuracy']]}")


def verdict(lif: dict, alif: dict) -> str:
    """按预注册判据返回结论文本。"""
    delta_pp = (alif["accuracy_mean"] - lif["accuracy_mean"]) * 100.0
    lines = [
        f"Δ（ALIF − LIF）= {delta_pp:+.2f} pp",
    ]

    unstable = [
        summary["label"]
        for summary in (lif, alif)
        if summary["accuracy_range_pp"] > STABILITY_RANGE_PP
    ]
    if unstable:
        lines.append(f"稳定性闸门触发（极差 > {STABILITY_RANGE_PP:.2f} pp）：{unstable}")
        lines.append("→ 只报区间，不报胜负。")
        return "\n".join(lines)

    divergent = [
        summary["label"]
        for summary in (lif, alif)
        if any(
            not math.isfinite(value) or value < 2 * RANDOM_BASELINE for value in summary["accuracy"]
        )
    ]
    if divergent:
        lines.append(f"发散闸门触发（有种子低于两倍随机基线）：{divergent}")
        lines.append("→ 结论记为发散，不报胜负。")
        return "\n".join(lines)

    if delta_pp > WIN_MARGIN_PP:
        lines.append(f"→ ALIF 在此预算下胜出（> +{WIN_MARGIN_PP:.2f} pp）。")
    elif delta_pp < -WIN_MARGIN_PP:
        lines.append(f"→ LIF 在此预算下胜出（> +{WIN_MARGIN_PP:.2f} pp）。")
    else:
        lines.append(f"→ 打平（|Δ| ≤ {WIN_MARGIN_PP:.2f} pp）；默认仍取 LIF（更便宜）。")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("sweep_results/neuron_models_pilot"),
        help="两臂 JSON 的父目录",
    )
    args = parser.parse_args(argv)

    lif = summarize(load_arm(args.root / "lif"), "LIF")
    alif = summarize(load_arm(args.root / "alif"), "ALIF")

    print("小参数模型：LIF vs ALIF（sMNIST，n_rec=64，10 epoch）\n")
    print_arm(lif)
    print_arm(alif)
    print()
    print(verdict(lif, alif))
    print()
    print("边界：10 epoch 是预算截断；sMNIST 不考核延迟记忆；只测了一个 ALIF 工作点。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
