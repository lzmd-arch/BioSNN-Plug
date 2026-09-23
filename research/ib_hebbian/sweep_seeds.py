"""W1 的多随机种子 runner：把验收跑重复 N 个种子并汇总。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

## 为什么需要它

`research/ib_hebbian/README.md` 的「已知边界」第 2 条写着：论文表 1/表 4 报的是 **5 个随机种子
的均值与极差**（MNIST 上极差约 0.3–0.4 个百分点），而本条目只报过 `base=0` 的单次运行。
**要报「与论文一致」，就得有同样的口径。** 这个脚本补上那一步。

它**不改训练逻辑**——每个种子仍然调用 :func:`research.ib_hebbian.train_mnist.train`，
只是把 `args.seed` 换掉、把结果收起来。所以单种子的复现记录仍然成立，多出来的只是同口径的统计量。

用法::

    uv run python -m research.ib_hebbian.sweep_seeds --seeds 0-4 --device cuda

每个种子落一个 JSON（`sweep_results/w1_seeds/`，与 W3 的扫描同一约定），最后打印均值与极差。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from research.common.device import select_device
from research.ib_hebbian.train_mnist import build_parser, train

__all__ = ["main", "parse_seeds"]

#: 论文表 1 / 表 4 在 MNIST 上的极差量级（百分点），用于把实测值放到同一把尺子上看。
PAPER_SPREAD_PP = 0.3


def parse_seeds(text: str) -> list[int]:
    """``"0-4"`` → ``[0, 1, 2, 3, 4]``；``"0,1,2"`` 也接受。

    Raises:
        ValueError: 格式不对，或区间反了。
    """
    seeds: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            low_text, _, high_text = chunk.partition("-")
            low, high = int(low_text), int(high_text)
            if high < low:
                raise ValueError(f"种子区间反了：{chunk!r}。")
            seeds.extend(range(low, high + 1))
        else:
            seeds.append(int(chunk))
    if not seeds:
        raise ValueError(f"没有解析出任何种子：{text!r}。")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"种子有重复：{text!r}。")
    return seeds


def build_parser_for_sweep() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seeds", default="0-4", help="种子，形如 0-4 或 0,1,2（默认 0-4）")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("sweep_results/w1_seeds"),
        help="每个种子一个 JSON 的落盘目录",
    )
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args, forwarded = build_parser_for_sweep().parse_known_args(argv)
    if "--smoke" in forwarded:
        raise SystemExit("多种子扫描不要用 --smoke：冒烟规模的精度没有意义。")

    seeds = parse_seeds(args.seeds)
    train_args = build_parser().parse_args(forwarded)
    device = select_device(args.device or train_args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"W1 多随机种子：{len(seeds)} 个种子（{seeds[0]}–{seeds[-1]}），设备 {device}")

    records = []
    for seed in seeds:
        train_args.seed = seed
        print(f"\n{'=' * 78}\n种子 {seed}\n{'=' * 78}", flush=True)
        started = time.perf_counter()
        accuracy = train(train_args, device)
        elapsed = time.perf_counter() - started
        record = {
            "seed": seed,
            "accuracy": accuracy,
            "elapsed_s": elapsed,
            "config": {
                "width": train_args.width,
                "n_layers": train_args.n_layers,
                "n_groups": train_args.n_groups,
                "epochs": train_args.epochs,
                "readout_kind": train_args.readout_kind,
                "readout_ridge": train_args.readout_ridge,
                "hidden_lr": train_args.hidden_lr,
                "batch_size": train_args.batch_size,
                "device": str(device),
            },
        }
        records.append(record)
        out_path = args.out_dir / f"seed{seed}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[落盘] {out_path}", flush=True)

    accuracies = [r["accuracy"] for r in records]
    spread_pp = (max(accuracies) - min(accuracies)) * 100
    print(f"\n{'=' * 78}")
    print(f"W1 多随机种子汇总（{len(seeds)} 个种子：{seeds[0]}–{seeds[-1]}）")
    print(f"{'=' * 78}")
    for record in records:
        print(
            f"  种子 {record['seed']:>3}  准确率 {record['accuracy']:.4f}  耗时 {record['elapsed_s']:.0f} s"
        )
    print(
        f"\n  均值 {statistics.fmean(accuracies):.4f}  最小 {min(accuracies):.4f}  "
        f"最大 {max(accuracies):.4f}  极差 {spread_pp:.2f} 个百分点"
    )
    print(f"  （论文表 1/表 4 在 MNIST 上的极差约 {PAPER_SPREAD_PP} 个百分点，可据此对照）")
    print(f"  JSON 落在 {args.out_dir}/，逐种子可查")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
