"""W2 的多随机种子 runner：把验收跑重复 N 个种子并汇总。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

## 为什么需要它

`research/eprop/README.md` 的「已知边界」第 3 条写着「**单次运行，不是多次平均。** 只有一个
种子（`base=0`）」。W1 与 W3 都有多种子口径（W1 对齐论文的 5 种子，W3 用 20 种子），
这一条缺一个同口径的统计量。

它**不改训练逻辑**——每个种子仍然调用
:func:`research.eprop.train_sequential.train`，只是换掉 `args.seed` 并收集结果。

## 记两件事，不只一件

W2 的验收判据有**两条**：测试准确率（"顺序任务可用"）与**活跃神经元比例 > 60%**（§七 的硬指标）。
所以每个种子同时记准确率与活跃比例，汇总时两条都给均值与极差——只报其中一条会让另一条
的种子方差看不见。

用法::

    uv run python -m research.eprop.sweep_seeds --seeds 0-4 --device cuda

每个种子落一个 JSON（`sweep_results/w2_seeds/`，与 W1/W3 的扫描同一约定），最后打印汇总。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from research.common.device import select_device
from research.eprop.train_sequential import build_parser, train

# 三条验证线只共享 research/common/，**互不 import**——所以这个解析器各自带一份
# （W3 的 sweep.py 也是这么做的）。

__all__ = ["main", "parse_seeds"]


def parse_seeds(spec: str) -> list[int]:
    """解析 ``"0-4"`` / ``"0,1,2"`` / ``"3"`` 形式的种子列表。

    Raises:
        ValueError: 格式不对、区间反了，或有重复（重复会让"均值"偷偷给某个种子加权）。
    """
    seeds: list[int] = []
    for chunk in spec.split(","):
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
        raise ValueError(f"没有解析出任何种子：{spec!r}。")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"种子有重复：{spec!r}。")
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
        default=Path("sweep_results/w2_seeds"),
        help="每个种子一个 JSON 的落盘目录",
    )
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    return parser


def _summary(values: list[float], label: str, criterion: str) -> str:
    mean = statistics.fmean(values)
    low, high = min(values), max(values)
    return (
        f"  {label:<12} 均值 {mean:.4f}  最小 {low:.4f}  最大 {high:.4f}  "
        f"极差 {(high - low) * 100:.2f} 个百分点   （判据：{criterion}）"
    )


def main(argv: list[str] | None = None) -> int:
    args, forwarded = build_parser_for_sweep().parse_known_args(argv)
    if "--smoke" in forwarded:
        raise SystemExit("多种子扫描不要用 --smoke：冒烟规模的精度没有意义。")

    seeds = parse_seeds(args.seeds)
    train_args = build_parser().parse_args(forwarded)
    device = select_device(args.device or train_args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"W2 多随机种子：{len(seeds)} 个种子（{seeds[0]}–{seeds[-1]}），"
        f"数据集 {train_args.dataset}，设备 {device}"
    )

    records = []
    for seed in seeds:
        train_args.seed = seed
        print(f"\n{'=' * 78}\n种子 {seed}\n{'=' * 78}", flush=True)
        started = time.perf_counter()
        accuracy, active, n_classes = train(train_args, device)
        elapsed = time.perf_counter() - started
        record = {
            "seed": seed,
            "test_accuracy": accuracy,
            "active_fraction": active,
            "n_classes": n_classes,
            "elapsed_s": elapsed,
            "config": {
                "dataset": train_args.dataset,
                "epochs": train_args.epochs,
                "n_rec": train_args.n_rec,
                "batch_size": train_args.batch_size,
                "beta": train_args.beta,
                "threshold": train_args.threshold,
                "learning_rate_in": train_args.learning_rate_in,
                "learning_rate_rec": train_args.learning_rate_rec,
                "learning_rate_out": train_args.learning_rate_out,
                "device": str(device),
            },
        }
        records.append(record)
        out_path = args.out_dir / f"seed{seed}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[落盘] {out_path}", flush=True)

    accuracies = [r["test_accuracy"] for r in records]
    actives = [r["active_fraction"] for r in records]
    print(f"\n{'=' * 78}")
    print(f"W2 多随机种子汇总（{len(seeds)} 个种子：{seeds[0]}–{seeds[-1]}，{train_args.dataset}）")
    print(f"{'=' * 78}")
    for record in records:
        print(
            f"  种子 {record['seed']:>3}  准确率 {record['test_accuracy']:.4f}  "
            f"活跃比例 {record['active_fraction']:.4f}  耗时 {record['elapsed_s']:.0f} s"
        )
    print()
    print(_summary(accuracies, "测试准确率", f"显著高于随机（{records[0]['n_classes']} 类）"))
    print(_summary(actives, "活跃神经元比例", "§七：「> 60%」"))
    print(f"  JSON 落在 {args.out_dir}/，逐种子可查")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
