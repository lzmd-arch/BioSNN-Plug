"""事件优先级 v2 的判据对账：配对比较 baseline 与 selective 的 10 个种子。

判据见 `sweep_results/event_priority_v2_prereg.md` 与 `_amend1.md`：

1. 10 种子均值相对 baseline 下降 ≤ 2.0pp（同一次运行里的配对）；
2. 实测选中时间步比例 ≤ 0.35；
3. 不出现 loss 非有限的训练发散。

用法::

    python scripts/analyze_event_priority_v2.py D:/Projects/v2_final6
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

__all__ = ["analyse", "main"]

DROP_MAX_PP = 2.0
FRACTION_MAX = 0.35
SEEDS_MIN = 10


def _load(directory: Path) -> dict[int, dict]:
    records = {}
    for path in sorted(directory.glob("seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records[int(payload["seed"])] = payload
    return records


def _finite_curve(record: dict) -> bool:
    curve = record.get("curve") or []
    for point in curve:
        for key in ("val_accuracy", "test_accuracy"):
            value = point.get(key)
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                return False
    accuracy = record.get("test_accuracy")
    return not (isinstance(accuracy, (int, float)) and not math.isfinite(float(accuracy)))


def analyse(root: Path) -> dict:
    baseline = _load(root / "baseline")
    selective = _load(root / "selective")
    seeds = sorted(set(baseline) & set(selective))
    if len(seeds) < SEEDS_MIN:
        raise SystemExit(f"配对种子不足：{len(seeds)} < {SEEDS_MIN}（{root}）")

    base_acc = [baseline[seed]["test_accuracy"] for seed in seeds]
    sel_acc = [selective[seed]["test_accuracy"] for seed in seeds]
    drops = [(base - selected) * 100.0 for base, selected in zip(base_acc, sel_acc, strict=True)]
    fractions = [float(selective[seed].get("mean_update_fraction", 1.0) or 1.0) for seed in seeds]
    finite = all(_finite_curve(selective[seed]) for seed in seeds) and all(
        _finite_curve(baseline[seed]) for seed in seeds
    )
    mean_drop = statistics.fmean(drops)
    criteria = {
        "baseline_mean": statistics.fmean(base_acc),
        "selective_mean": statistics.fmean(sel_acc),
        "mean_drop_pp": mean_drop,
        "median_drop_pp": statistics.median(drops),
        "criterion_1_drop": mean_drop <= DROP_MAX_PP,
        "fraction_mean": statistics.fmean(fractions),
        "criterion_2_fraction": statistics.fmean(fractions) <= FRACTION_MAX,
        "criterion_3_finite": finite,
    }
    criteria["ready"] = (
        criteria["criterion_1_drop"]
        and criteria["criterion_2_fraction"]
        and criteria["criterion_3_finite"]
    )
    return {
        "root": str(root),
        "seeds": seeds,
        "baseline_accuracy": base_acc,
        "selective_accuracy": sel_acc,
        "drop_pp": drops,
        "selected_fraction": fractions,
        "best_steps": {
            "baseline": [baseline[seed].get("best_steps") for seed in seeds],
            "selective": [selective[seed].get("best_steps") for seed in seeds],
        },
        "criteria": criteria,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="事件优先级 v2 判据对账")
    parser.add_argument("root", type=Path, help="包含 baseline/ 与 selective/ 的目录")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    summary = analyse(args.root)
    criteria = summary["criteria"]
    print(
        f"baseline {criteria['baseline_mean']:.4f} vs selective {criteria['selective_mean']:.4f}；"
        f"配对下降 均值 {criteria['mean_drop_pp']:+.2f}pp（中位 {criteria['median_drop_pp']:+.2f}pp）；"
        f"选中比例 {criteria['fraction_mean']:.3f}；发散 {not criteria['criterion_3_finite']} → "
        f"{'ready' if criteria['ready'] else 'not ready'}"
    )
    for seed, base, sel, drop, fraction in zip(
        summary["seeds"],
        summary["baseline_accuracy"],
        summary["selective_accuracy"],
        summary["drop_pp"],
        summary["selected_fraction"],
        strict=True,
    ):
        print(f"  种子 {seed:>2}: {base:.4f} → {sel:.4f}（{drop:+.2f}pp），选中 {fraction:.3f}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
