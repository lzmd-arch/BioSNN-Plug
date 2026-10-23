"""协同性基准的结果汇总：把各臂的 ``summary.json`` 收成预注册的那几个数。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

判据见 [`sweep_results/arbiter_synergy_task_spec.md`](../../sweep_results/arbiter_synergy_task_spec.md)
的「判据改写」一节。这里只做算术，**不改判据**：

| 项 | 定义 | 门限 |
| :--- | :--- | :--- |
| 门禁 | 每次运行 ``‖ΔW‖/‖W₀‖`` | ≥ 0.05，否则该次运行作废、不进统计 |
| 协同量 | ``Fisher(MIX) − max(Fisher(H), Fisher(E))``，**按种子成对** | 中位数 ≥ δ = 0.05 |
| 稳定性 | 5 个种子的四分位区间 | MIX 的**下四分位** > H/E 的**上四分位** |

用法::

    uv run python -m research.arbiter.summarise_synergy --root sweep_results/arbiter_synergy

目录布局按 runner 的 ``--out``：一个目录一套配置，里面是 ``summary.json``。

⚠️ **作废的运行必须报出来**。门禁 FAIL 的种子从统计里剔除，但数量要写在输出里——
悄悄丢掉会让"5 个种子"变成"3 个种子的结论"。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

__all__ = ["GATE_RATIO", "SYNERGY_DELTA", "load_groups", "quartiles", "summarise"]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "sweep_results" / "arbiter_synergy"

#: 与 runner 的门禁一致；统计里再判一次，防止把别处产出的 json 混进来。
GATE_RATIO = 0.05

#: 协同量门限 δ。**在看到 MIX 数字之前定的**（见规格）。
SYNERGY_DELTA = 0.05


def load_groups(root: Path) -> dict[str, list[dict]]:
    """``root`` 下每个子目录读一份 ``summary.json``，返回 ``{目录名: rows}``。"""
    groups: dict[str, list[dict]] = {}
    for summary in sorted(Path(root).glob("*/summary.json")):
        payload = json.loads(summary.read_text(encoding="utf-8"))
        groups[summary.parent.name] = payload["rows"]
    return groups


def quartiles(values: list[float]) -> tuple[float, float, float]:
    """``(下四分位, 中位数, 上四分位)``。样本少于 4 个时用线性插值，结论要谨慎。"""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.percentile(array, 25)),
        float(np.median(array)),
        float(np.percentile(array, 75)),
    )


def _fisher_by_seed(rows: list[dict]) -> tuple[dict[int, float], list[int]]:
    """按种子取 Fisher；门禁 FAIL 的种子剔除并单独返回。"""
    kept: dict[int, float] = {}
    dropped: list[int] = []
    for row in rows:
        seed = int(row["seed"])
        if not row.get("gate_passed", False):
            dropped.append(seed)
            continue
        kept[seed] = float(row["fisher"])
    return kept, dropped


def summarise(groups: dict[str, list[dict]], *, delta: float = SYNERGY_DELTA) -> dict:
    """按规格里的三条判据汇总。返回的是**可直接写进结果文档**的结构。"""
    per_group: dict[str, dict] = {}
    for name, rows in groups.items():
        kept, dropped = _fisher_by_seed(rows)
        if kept:
            q1, median, q3 = quartiles(list(kept.values()))
        else:
            q1 = median = q3 = float("nan")
        per_group[name] = {
            "seeds_used": sorted(kept),
            "seeds_dropped": sorted(dropped),
            "median": median,
            "q1": q1,
            "q3": q3,
            "fisher_by_seed": kept,
        }

    result: dict = {"per_group": per_group, "delta": delta}

    try:
        hebbian = per_group["H"]["fisher_by_seed"]
        eprop = per_group["E"]["fisher_by_seed"]
        mix = per_group["MIX"]["fisher_by_seed"]
    except KeyError:
        result["criterion"] = {"status": "缺少 H / E / MIX 中至少一组，无法判定"}
        return result

    seeds = sorted(set(hebbian) & set(eprop) & set(mix))
    synergy = {seed: mix[seed] - max(hebbian[seed], eprop[seed]) for seed in seeds}
    q1, median, q3 = quartiles(list(synergy.values()))

    # ⚠️ 分离判据比较的是 **MIX 的 Fisher 下四分位** 与 H/E 的上四分位，
    # 不是协同量的四分位。第一版写成了后者：协同量 q1 ≈ 0.175 被拿去和臂的 q3 ≈ 0.472 比，
    # 判据必然为假——两个量纲都不是一回事。
    mix_q1 = per_group["MIX"]["q1"]
    best_arm_q3 = max(per_group["H"]["q3"], per_group["E"]["q3"])
    delta_ok = bool(np.isfinite(median) and median >= delta)
    separation_ok = bool(np.isfinite(mix_q1) and np.isfinite(best_arm_q3) and mix_q1 > best_arm_q3)

    result["synergy"] = {
        "per_seed": {seed: float(value) for seed, value in synergy.items()},
        "seeds_used": seeds,
        "q1": q1,
        "median": median,
        "q3": q3,
    }
    result["criterion"] = {
        "delta_ok": delta_ok,
        "separation_ok": separation_ok,
        "passed": bool(delta_ok and separation_ok),
        "mix_q1": mix_q1,
        "best_arm_q3": best_arm_q3,
    }

    # 对照：冻结 α 与打乱标签。缺了就明说缺，不假装通过。
    for control in ("MIX_frozen", "MIX_shuffle"):
        if control in per_group:
            control_median = per_group[control]["median"]
            result.setdefault("controls", {})[control] = {
                "median": control_median,
                "vs_mix": float(control_median - per_group["MIX"]["median"]),
            }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="协同性基准结果汇总")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json", type=Path, help="把汇总写到这个文件（默认只打印）")
    args = parser.parse_args(argv)

    groups = load_groups(args.root)
    if not groups:
        raise SystemExit(f"{args.root} 下没有找到 */summary.json")
    result = summarise(groups)

    for name, stats in result["per_group"].items():
        print(
            f"{name:12s} 种子 {stats['seeds_used']}  作废 {stats['seeds_dropped']}  "
            f"Fisher 中位数 {stats['median']:.3f}  [{stats['q1']:.3f}, {stats['q3']:.3f}]"
        )
    synergy = result.get("synergy")
    if synergy:
        print(
            f"\n协同量（按种子成对）：中位数 {synergy['median']:+.3f}  "
            f"[{synergy['q1']:+.3f}, {synergy['q3']:+.3f}]  种子 {synergy['seeds_used']}"
        )
        for seed, value in sorted(synergy["per_seed"].items()):
            print(f"  seed {seed}: {value:+.3f}")
    criterion = result["criterion"]
    print(f"\n判据：{criterion}")
    for name, stats in result.get("controls", {}).items():
        print(f"对照 {name}: 中位数 {stats['median']:.3f}（相对 MIX {stats['vs_mix']:+.3f}）")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"\nwritten to {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
