"""多随机种子扫描：把"种子方差"当成一等公民来测。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

## 为什么需要它

CartPole 这条线上的**种子方差极大**：同一配置内实测能差 3.5 倍（零均值痕迹变体逐种子
500 / 43.8 / 73.6 / 100.1 / 111）。由此产生过两个具体的失败：

1. 只用 1–3 个种子得出的"改善"是噪声。零均值变体的五种子均值 +89% 几乎全由 seed 0 一个
   离群点撑起，中位数只 +13.4，去掉那个种子后与基线等价。
2. 有人的扫描脚本用多个并发进程追加同一个日志文件，而 Windows 的 append **不是原子**的
   （``_O_APPEND`` 要先 seek 到文件尾再写），30 次运行只落地 6 行，其中一行还内部矛盾。
   那份证据整批作废。

所以这个 runner 的两条硬规矩：**顺序执行**、**每次运行写自己的文件**。

## 统计口径

判据用**中位数**——种子分布明显右偏，均值会被离群点带走。同时**必报最小值与最大值**：
§七 要求的是"**稳定**学到 200 步以上"，单种子满分不算达标，最小值才说明稳定性。

## 用法

::

    # 列出要跑哪些格子，不真跑（先把"要跑什么"摆出来）
    uv run python -m research.rstdp.sweep --grid encoding_sigma=0.5,1.0 --seeds 0-9 --dry-run

    # 真跑：2 × 10 = 20 次，每次约 4 秒
    uv run python -m research.rstdp.sweep --grid encoding_sigma=0.5,1.0 --seeds 0-9

    # 多轴交叉
    uv run python -m research.rstdp.sweep \\
        --grid encoding_sigma=0.5,1.0 --grid n_features=64,256 --seeds 0-9

    # 开关类参数用 真/假
    uv run python -m research.rstdp.sweep --grid actor_normalize=true,false --seeds 0-9

**未跑或失败的格子会被显式打印出来**，不静默截断——"看起来跑完了"与"真跑完了"必须能分辨。
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

from research.common.device import select_device
from research.rstdp.cartpole import TrialResult, run_trial

__all__ = ["expand_grid", "parse_seeds", "summarize"]

#: 计划书 §七 的验收判据。
ACCEPTANCE_STEPS = 200.0
RUNS_PER_SECOND = 0.03  # 单次约 33 秒。**这个数随配置差得很远**：策略差时一回合只有几十步、
# 整次几秒；策略好时一回合跑满 500 步、整次可以到一分钟以上。它只用于给量级，不是承诺。


def parse_seeds(spec: str) -> list[int]:
    """解析 ``"0-9"`` / ``"0,1,2"`` / ``"5"`` 形式的种子列表。"""
    seeds: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            low, high = chunk.split("-", 1)
            start, stop = int(low), int(high)
            if stop < start:
                raise ValueError(f"种子区间颠倒：{chunk!r}")
            seeds.extend(range(start, stop + 1))
        else:
            seeds.append(int(chunk))
    if not seeds:
        raise ValueError(f"没有解析出任何种子：{spec!r}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"种子列表有重复：{spec!r}")
    return seeds


def _coerce(raw: str):
    """把命令行字符串转成合适的 Python 值（bool / int / float / str）。"""
    lowered = raw.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return raw


def expand_grid(grid_specs: list[str]) -> list[dict]:
    """把 ``--grid`` 的多个轴展开成配置的**叉积**。

    Raises:
        ValueError: 轴名不是 :func:`~research.rstdp.cartpole.run_trial` 的参数。
    """
    import inspect

    allowed = set(inspect.signature(run_trial).parameters) - {"seed", "device", "verbose"}
    axes: dict[str, list] = {}
    for spec in grid_specs:
        if "=" not in spec:
            raise ValueError(f"--grid 需要 key=v1,v2 的形式，收到 {spec!r}")
        key, values = spec.split("=", 1)
        key = key.strip()
        if key not in allowed:
            raise ValueError(
                f"--grid 的键 {key!r} 不是 run_trial 的参数。可用的有：{sorted(allowed)}"
            )
        axes[key] = [_coerce(v) for v in values.split(",") if v.strip()]

    if not axes:
        return [{}]
    keys = list(axes)
    return [
        dict(zip(keys, combo, strict=True)) for combo in itertools.product(*(axes[k] for k in keys))
    ]


def summarize(values: list[float]) -> dict[str, float]:
    """一组逐种子值的统计量。

    中位数是**主判据**（分布右偏，均值会被离群点带走）；``spread`` = 最大/最小，
    用来一眼看出种子方差有多大——本问题上它经常大于 3。
    """
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "spread": (ordered[-1] / ordered[0]) if ordered[0] > 0 else float("inf"),
    }


def config_slug(config: dict) -> str:
    if not config:
        return "default"
    return "_".join(f"{k}{v}" for k, v in sorted(config.items()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        help="配置轴，形如 key=v1,v2（可重复；多个轴取叉积）",
    )
    parser.add_argument("--seeds", default="0-4", help="种子，形如 0-9 或 0,1,2（默认 0-4）")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("sweep_results"),
        help="逐次运行的结果落盘目录（每次运行一个文件，避免写竞争）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只列出要跑的格子，不真跑")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="固定在哪个设备上跑。**默认 cpu**：验收基线是在 CPU 上记录的，"
        "而 CPU 与 CUDA 的浮点与随机流不同，实测同一配置同一种子的结果并不相等，"
        "所以换设备等于换了一套数字，不能混着比",
    )
    parser.add_argument("--episodes", type=int, default=None, help="覆盖训练回合数")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = parse_seeds(args.seeds)
    configs = expand_grid(args.grid)
    cells = [(c, s) for c in configs for s in seeds]
    estimated = len(cells) / RUNS_PER_SECOND

    print(f"配置 {len(configs)} 个 × 种子 {len(seeds)} 个 = **{len(cells)} 次运行**")
    print(f"单次约 {1 / RUNS_PER_SECOND:.0f} 秒，顺序执行，预计 {estimated / 60:.0f} 分钟")
    print(f"逐次结果落盘到 {args.out_dir}/（每次运行一个文件）\n")

    if args.dry_run:
        print("要跑的格子：")
        for index, (config, seed) in enumerate(cells, 1):
            print(f"  {index:3d}. {config_slug(config):40s} seed={seed}")
        print("\n（--dry-run：一次都没跑）")
        return 0

    device = select_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[TrialResult]] = {}
    failures: list[tuple[dict, int, str]] = []

    for index, (config, seed) in enumerate(cells, 1):
        slug = config_slug(config)
        try:
            # 先铺默认值再让 config 覆盖：否则 --grid episodes=... 会与这里的显式
            # episodes 撞车（got multiple values for keyword argument）。
            kwargs = {"episodes": args.episodes or 800, "device": device}
            kwargs.update(config)
            result = run_trial(seed, **kwargs)
        except Exception as exc:
            failures.append((config, seed, f"{type(exc).__name__}: {exc}"))
            print(f"[{index:3d}/{len(cells)}] {slug} seed={seed}  ✗ 失败：{exc}", flush=True)
            continue

        results.setdefault(slug, []).append(result)
        payload = result.__dict__ | {"config": config}
        (args.out_dir / f"{index:03d}_{slug}_seed{seed}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        peak = (
            f"  峰值 {result.peak_mean_steps:6.1f} @{result.peak_episode}" if result.curve else ""
        )
        print(
            f"[{index:3d}/{len(cells)}] {slug} seed={seed}  → {result.mean_steps:6.1f} 步"
            f"（{result.min_steps}–{result.max_steps}）  偏移比 {result.offset_ratio:.4f}{peak}",
            flush=True,
        )

    _print_summary(results, seeds)
    if failures:
        print(
            f"\n⚠️ {len(failures)} 个格子没有产出数据——注意这与「跑出来是 0」不同：它是**没跑成**："
        )
        for config, seed, why in failures:
            print(f"  {config_slug(config)} seed={seed}：{why}")
    else:
        print(f"\n全部 {len(cells)} 个格子都有数据。")
    return 1 if failures else 0


def _print_summary(results: dict[str, list[TrialResult]], seeds: list[int]) -> None:
    print("\n" + "=" * 96)
    show_peak = any(r.curve for rows in results.values() for r in rows)
    header = f"{'配置':<34}{'n':>3}{'中位数':>10}{'均值':>10}{'最小':>8}{'最大':>8}{'极差':>8}{'偏移比':>10}"
    print(header + ("{:>12}".format("峰值中位数") if show_peak else ""))
    print("-" * (len(header) + (12 if show_peak else 0)))
    for slug, rows in sorted(
        results.items(), key=lambda kv: -summarize([r.mean_steps for r in kv[1]]).get("median", 0)
    ):
        stats = summarize([r.mean_steps for r in rows])
        offsets = [r.offset_ratio for r in rows]
        line = (
            f"{slug:<34}{stats['n']:>3}{stats['median']:>10.1f}{stats['mean']:>10.1f}"
            f"{stats['min']:>8.0f}{stats['max']:>8.0f}{stats['spread']:>8.1f}"
            f"{statistics.median(offsets):>10.4f}"
        )
        if show_peak:
            peaks = summarize([r.peak_mean_steps for r in rows if r.curve])
            line += f"{peaks.get('median', float('nan')):>12.1f}"
        print(line)
    print("-" * (len(header) + (12 if show_peak else 0)))
    if show_peak:
        print(
            "「峰值中位数」是训练途中贪心评测的最好一次，**只作诊断**：它把「搜索不到好解」与"
            "「找到了但留不住」分开，这两者的修法不相交。验收数字仍取最终权重。"
        )
    print(f"判据（计划书 §七）：中位数 >= {ACCEPTANCE_STEPS:.0f} 步")
    passed = [
        s
        for s, rows in results.items()
        if summarize([r.mean_steps for r in rows]).get("median", 0) >= ACCEPTANCE_STEPS
    ]
    print(f"达标的配置：{passed if passed else '无'}")
    print("注意「极差」一列：本问题上它经常 > 3，说明单种子结论不可用。")


if __name__ == "__main__":
    raise SystemExit(main())
