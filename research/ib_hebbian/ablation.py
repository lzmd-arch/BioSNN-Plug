"""W1 的消融对照：每一条臂对应论文 Table 1/表 3 里的**具体某一列**。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

## 为什么是这几条

`research/ib_hebbian/README.md` 的「已知边界」第 5 条写着：消融只做了除法归一化一条、且是弱形式，
**HSIC vs pHSIC、余弦核 vs 高斯核、分组 vs 无分组尚未复现**。这个 runner 把它们固定下来。

## 每条臂对应论文的哪一列（考证自 arXiv:2006.07123v2 的 Table 3 与官方实现）

| 臂 | 开关 | 论文里的列（MNIST 行） |
| :--- | :--- | :--- |
| `baseline` | 全部默认 | `pHSIC: Gaussian` + `grp+div`（η_l=1.0、c^k=32）——**就是本仓库的验收配置** |
| `cossim` | `--kernel cossim` | `pHSIC: cossim` + `grp+div`（η_l=0.4、c^k=16） |
| `divnorm_off` | `--divnorm off` | `pHSIC: Gaussian` + `grp`（η_l=1.0、c^k=32） |
| `grouping_off` | `--grouping off --divnorm off` | `pHSIC: Gaussian` plain（η_l=0.6） |
| `hsic` | `--hsic-estimate-mode biased` | **论文没有这一列的数**：§5.1 与附录 D.8 只给定性结论（"didn't improve performance" / "not shown"） |

⚠️ 三个坑（都是考证里查实的，别用别的写法代替）：
- **「无分组」不是 `n_groups=1`**：那样目标恒为 0、梯度恒为 0，层照跑但永不学习且不报错。
  用 `--grouping off`（活动直接进核）。本仓库已对这个陷阱加了护栏。
- **「无除法归一化」不是 `--divnorm-power 0`**：那个模块在 power=0 时仍做组内居中、返回 ∘z 而非 z。
  用 `--divnorm off`（旁路整个模块）。
- 论文的 plain 列**同时**没有分组、也没有除法归一化，所以 `grouping_off` 这一臂要两个开关一起关。

## 超参对不上论文的地方（读数字时要记得）

臂与臂之间**只换那一个开关**，其余超参沿用本仓库的验收配置（η_l=1.0、c_k=32、σ=5、γ=2），
**没有**按论文每一列各自的 η_l/c^k 重调。所以与论文对应列的比较是「同一开关语义下」的比较，
不是逐点复现——这一点写进 README 的已知边界。

用法::

    uv run python -m research.ib_hebbian.ablation                    # 全部臂，种子 0
    uv run python -m research.ib_hebbian.ablation --arms baseline,hsic --seeds 0-2
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from research.common.device import select_device
from research.ib_hebbian.sweep_seeds import parse_seeds
from research.ib_hebbian.train_mnist import build_parser, train

__all__ = ["ARMS", "main"]

#: 消融臂：名字 → 对 `train_mnist` 命令行参数的覆盖。
#: `baseline` 是**空覆盖**，跑出来的应当与验收配置逐字一致——这本身就是一次自检：
#: 加了一堆开关之后，默认路径没有被改动。
ARMS: dict[str, dict[str, object]] = {
    "baseline": {},
    "cossim": {"kernel": "cossim"},
    "divnorm_off": {"divnorm": "off"},
    "grouping_off": {"grouping": "off", "divnorm": "off"},
    "hsic": {"hsic_estimate_mode": "biased"},
}

#: 每条臂在论文里对应的列（写进 JSON，免得以后对不上号）。
PAPER_COLUMN = {
    "baseline": "Table 3, MNIST: pHSIC Gaussian + grp+div（η_l=1.0, c^k=32）",
    "cossim": "Table 3, MNIST: pHSIC cossim + grp+div（η_l=0.4, c^k=16）",
    "divnorm_off": "Table 3, MNIST: pHSIC Gaussian + grp（η_l=1.0, c^k=32）",
    "grouping_off": "Table 3, MNIST: pHSIC Gaussian plain（η_l=0.6）",
    "hsic": "**论文没有这一列的数字**（§5.1 / 附录 D.8 只有定性结论）",
}


def build_parser_for_ablation() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--arms",
        default=",".join(ARMS),
        help=f"要跑哪些臂，逗号分隔（可用：{', '.join(ARMS)}；默认全部）",
    )
    parser.add_argument("--seeds", default="0", help="每个臂跑哪些种子，形如 0-2（默认 0）")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("sweep_results/w1_ablation"),
        help="每个臂 × 每个种子一个 JSON 的落盘目录",
    )
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args, forwarded = build_parser_for_ablation().parse_known_args(argv)

    names = [name.strip() for name in args.arms.split(",") if name.strip()]
    unknown = [name for name in names if name not in ARMS]
    if unknown:
        raise SystemExit(f"未登记的臂：{unknown}。可用：{list(ARMS)}。")
    if "--smoke" in forwarded:
        raise SystemExit("消融对照不要用 --smoke：冒烟规模的精度没有意义。")

    seeds = parse_seeds(args.seeds)
    train_args = build_parser().parse_args(forwarded)
    device = select_device(args.device or train_args.device)
    # **覆盖前的原始值要先留一份**：`train_args` 会被逐臂改写，改写后就拿不到默认值了。
    touched = sorted({key for arm in ARMS.values() for key in arm})
    defaults = {key: getattr(train_args, key) for key in touched}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"W1 消融：{len(names)} 个臂 × {len(seeds)} 个种子，设备 {device}")

    results: dict[str, list[float]] = {}
    for name in names:
        results[name] = []
        for seed in seeds:
            for key in touched:  # 先复位，免得上一条臂的覆盖串味
                setattr(train_args, key, defaults[key])
            for key, value in ARMS[name].items():
                setattr(train_args, key, value)
            train_args.seed = seed

            setting = ", ".join(f"{k}={v}" for k, v in ARMS[name].items()) or "（验收配置，无覆盖）"
            print(f"\n{'=' * 78}\n臂 {name}（{setting}）种子 {seed}\n{'=' * 78}", flush=True)
            started = time.perf_counter()
            accuracy = train(train_args, device)
            elapsed = time.perf_counter() - started
            results[name].append(accuracy)
            record = {
                "arm": name,
                "overrides": ARMS[name],
                "paper_column": PAPER_COLUMN[name],
                "seed": seed,
                "accuracy": accuracy,
                "elapsed_s": elapsed,
            }
            out_path = args.out_dir / f"{name}_seed{seed}.json"
            out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[落盘] {out_path}", flush=True)

    baseline = statistics.fmean(results["baseline"]) if "baseline" in results else None
    print(f"\n{'=' * 78}")
    print("W1 消融对照" + ("（基线为 baseline 臂）" if baseline is not None else ""))
    print(f"{'=' * 78}")
    for name in names:
        values = results[name]
        mean = statistics.fmean(values)
        line = f"  {name:<14} 均值 {mean:.4f}"
        if len(values) > 1:
            line += f"  最小 {min(values):.4f}  最大 {max(values):.4f}"
        if baseline is not None and name != "baseline":
            line += f"   Δ vs baseline {100 * (mean - baseline):+.2f} 个百分点"
        print(line)
    print(f"\n  JSON 落在 {args.out_dir}/，逐臂逐种子可查（每条臂都记了它对应论文的哪一列）")
    print("  提醒：hsic 那一臂**没有论文数字可比**——论文只给了定性结论（§5.1 / 附录 D.8）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
