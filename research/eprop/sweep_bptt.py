"""BPTT 臂的多种子 runner：与 e-prop 的 ``sweep_seeds`` **同预算、同种子**。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

## 为什么不是复用 ``bptt_baseline.py``

``bptt_baseline.py`` 产出的**不是**这条臂，而是那份「量化差距报告」artifact：它一次跑
两条臂、10 epoch、2 万样本，CI 的 ``gap-report`` job 依赖它。拿它去对 e-prop 的收敛值
（0.8831）会同时错三件事：样本数不同（2 万 vs 5.4 万）、epoch 不同（10 vs 250）、
**没有多种子**（那边的 0.8831 是 3 个种子的均值）。所以本模块是 ``sweep_seeds.py``
在 BPTT 侧的对应物，只借 ``BPTTLearner``，不碰那条已记录的路径。

## 配对设计是真的，不只是「同 seed」

:func:`~research.common.seeding.derive_seed` 按**用途名的 crc32** 派生（与调用顺序无关），
所以只要用同一个 ``SeedBook`` 基准与同一套用途名，这里拿到的

* 训练/验证集划分（``划分验证集``）
* 初始权重（``权重初始化``）
* 每批打乱顺序（``每批打乱``）
* 脉冲编码（``脉冲编码``）

与 e-prop 那次运行**逐位相同**。两条臂的差别因此只剩学习规则本身——这正是论文
「同架构/同损失/同预算/同 seed，唯一差别是学习规则」那句话要的东西。

## 公平性：BPTT 有自己的 lr

只跑一个 lr 会**低估** BPTT，于是头条数字会被读成「e-prop 赢了」而不是「同最优条件下
的差距」。所以流程是两段：先用 30 epoch × 3 档 lr 选出 BPTT 自己的最优 lr，再用那个
lr 跑满预算。选 lr 的那一段的产物也要留档——「我们怎么选的」本身是方法的一部分。

## 已知的不对称（记录在案，没有偷偷抹平）

:class:`~research.eprop.bptt_baseline.BPTTLearner` 用**单个** ``SGD(momentum=0.9)``
覆盖全部参数，读出层拿不到 e-prop 那样的独立 ``eta_out=5e-2``。这条不对称对 BPTT 是
**不利**的。它不在本轮授权范围内（改它就等于改那条已记录 artifact 的含义），所以照原样
跑并写进备注；读这个数字的人应该知道它。

用法::

    uv run python -m research.eprop.sweep_bptt --seeds 0-2 --epochs 30 \\
        --learning-rate 3e-3 --out-dir sweep_results/w2_bptt_lr003

    uv run python -m research.eprop.sweep_bptt --smoke   # 只验管线，数字无意义
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from research.common.device import describe_device, measure_peak_memory, select_device
from research.common.provenance import collect
from research.common.seeding import SeedBook
from research.eprop.bptt_baseline import BPTTLearner
from research.eprop.neurons import ALIFState
from research.eprop.sweep_seeds import parse_seeds
from research.eprop.tasks import load_shd, load_smnist

__all__ = ["apply_smoke", "build_parser", "run_seed"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dataset", default="smnist", choices=["smnist", "shd"])
    parser.add_argument("--seeds", default="0-2", help="种子，形如 0-2 或 0,1,2（默认 0-2）")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("sweep_results/w2_bptt"),
        help="每个种子一个 JSON 的落盘目录",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-rec", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.07, help="ALIF 适应强度（0 = LIF）")
    parser.add_argument("--threshold", type=float, default=0.62)
    parser.add_argument("--tau-adaptation", type=float, default=500.0)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-3,
        help="BPTT 的 lr（单个优化器覆盖全部参数，见模块 docstring 的「已知的不对称」）",
    )
    parser.add_argument("--decay-out", type=float, default=0.95)
    parser.add_argument(
        "--n-train",
        type=int,
        default=0,
        help="训练样本数上限；0 = 全量减去验证集。**非 0 会破坏与 e-prop 的预算匹配**，"
        "只给冒烟/调试用",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="每几轮评一次验证集。默认 1，与 sweep_seeds 强行加上的那个值一致——"
        "曲线是论文图的数据源，口径要跟 e-prop 那边一样",
    )
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument(
        "--smoke", action="store_true", help="只验管线：1 epoch / 1 种子 / 缩小规模"
    )
    return parser


@torch.no_grad()
def _spike_stats(model: BPTTLearner, inputs: torch.Tensor) -> tuple[float, float]:
    """活跃神经元比例与脉冲稀疏度。

    :class:`BPTTLearner` 的前向只返回 logits、不留脉冲，所以这里按它的前向循环重跑
    一遍（无梯度）。两个量的定义与 ``EPropLearner`` 的同名方法逐行一致，否则两条臂的
    这一列不可比。
    """
    state = ALIFState.zeros(
        inputs.shape[1], model.cell.n_rec, device=inputs.device, dtype=inputs.dtype
    )
    spikes = []
    for t in range(inputs.shape[0]):
        z, state = model.cell.step(inputs[t], state)
        spikes.append(z)
    stacked = torch.stack(spikes)
    fired = stacked.reshape(-1, model.cell.n_rec) > 0
    active = float(fired.any(dim=0).float().mean().item())
    sparsity = float((stacked == 0).float().mean().item())
    return active, sparsity


def run_seed(args: argparse.Namespace, seed: int, device: torch.device) -> dict:
    """跑一个种子，返回落盘的那条记录（键与 ``sweep_seeds`` 对齐）。"""
    book = SeedBook(base=seed)
    book.apply("全局种子")

    loader = load_shd if args.dataset == "shd" else load_smnist
    train_set = loader(args.data_dir, split="train", seed=book.derive("脉冲编码"))
    test_set = loader(args.data_dir, split="test", seed=book.derive("脉冲编码"))
    n_in = int(train_set.inputs.shape[-1])
    n_classes = int(max(int(train_set.labels.max()), int(test_set.labels.max()))) + 1

    # 与 train_sequential 逐行同构：10% 作验证集，剩下的才训练。默认（``--n-train 0``）下
    # 样本数自动是「全量减验证集」——**预算由构造保证**，不靠调用方记得传对；``--n-train``
    # 非 0 只为冒烟/调试存在，它一被用上，这个数字就不再能与 e-prop 那次相减。
    n_samples = min(len(train_set), args.n_train) if args.n_train else len(train_set)
    n_val = max(n_samples // 10, 1)
    generator = torch.Generator().manual_seed(book.derive("划分验证集"))
    order = torch.randperm(len(train_set), generator=generator)[:n_samples]
    val_index, train_index = order[:n_val], order[n_val:]
    train_inputs, train_labels = train_set.batch(train_index)
    val_inputs, val_labels = train_set.batch(val_index)
    test_inputs, test_labels = test_set.batch(torch.arange(len(test_set)))
    if args.n_train:
        # 冒烟时测试集也缩，否则 CPU 上光评测就比训练还久（与 train_sequential 的冒烟同构）。
        keep = max(args.n_train // 4, 32)
        test_inputs, test_labels = test_inputs[:, :keep], test_labels[:keep]

    train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

    model = BPTTLearner(
        n_in,
        args.n_rec,
        n_classes,
        tau_adaptation=args.tau_adaptation,
        threshold=args.threshold,
        beta=args.beta,
        learning_rate=args.learning_rate,
        decay_out=args.decay_out,
        generator=torch.Generator().manual_seed(book.derive("权重初始化")),
    ).to(device)

    shuffle = torch.Generator().manual_seed(book.derive("每批打乱"))
    started = time.perf_counter()
    n_train = len(train_labels)
    history: list[dict[str, float]] = []

    with measure_peak_memory(device) as memory:
        for epoch in range(1, args.epochs + 1):
            permutation = torch.randperm(n_train, generator=shuffle)
            running = 0.0
            batches = 0
            for start in range(0, n_train, args.batch_size):
                chunk = permutation[start : start + args.batch_size]
                running += model.update(train_inputs[:, chunk], train_labels[chunk])
                batches += 1
            running /= max(batches, 1)

            if epoch % args.eval_every == 0 or epoch == args.epochs:
                val_accuracy = float(
                    (model.predict(val_inputs) == val_labels).float().mean().item()
                )
                print(
                    f"epoch {epoch:3d}  train_ce {running:.4f}  val_acc {val_accuracy:.4f}",
                    flush=True,
                )
                history.append({"epoch": float(epoch), "val_accuracy": val_accuracy})

    elapsed = time.perf_counter() - started

    test_accuracy = float((model.predict(test_inputs) == test_labels).float().mean().item())
    active, sparsity = _spike_stats(model, test_inputs)

    print(f"\n测试准确率：{test_accuracy:.4f}（{n_classes} 类随机基线 {1 / n_classes:.2f}）")
    print(f"活跃神经元比例：{active:.4f}")
    print(f"脉冲稀疏度：{sparsity:.4f}")

    record = collect(
        f"eprop/bptt-{args.dataset}",
        seeds=book.render(),
        elapsed_s=elapsed,
        gpu=describe_device(device).render(),
        peak_mb=memory["peak_mb"],
        notes=[
            f"BPTT 臂（对照 e-prop 同种子运行）：n_rec={args.n_rec}, beta={args.beta}, "
            f"thr={args.threshold}, eta={args.learning_rate}, decay_out={args.decay_out}, "
            f"epochs={args.epochs}, batch={args.batch_size}",
            "预算由构造保证：训练集同样是从全量里切掉 10% 验证集，"
            "划分/初始化/打乱/编码四处的种子与 e-prop 那次逐位相同（用途名派生，与顺序无关）",
            "已知不对称（对 BPTT 不利，未抹平）：BPTTLearner 用单个 SGD(momentum=0.9) "
            "覆盖全部参数，读出层没有 e-prop 那样的独立 eta_out=5e-2",
            "代理梯度穿过脉冲函数；与 e-prop 的差别只在学习规则",
        ],
    )
    print("\n" + record.render_block())

    return {
        "seed": seed,
        "test_accuracy": test_accuracy,
        "active_fraction": active,
        "spike_sparsity": sparsity,
        "n_classes": n_classes,
        "elapsed_s": elapsed,
        "curve": history,
        "config": {
            "dataset": args.dataset,
            "epochs": args.epochs,
            "n_rec": args.n_rec,
            "batch_size": args.batch_size,
            "beta": args.beta,
            "threshold": args.threshold,
            "learning_rate": args.learning_rate,
            "decay_out": args.decay_out,
            "n_train": n_train,
            "n_val": n_val,
            "device": str(device),
        },
    }


def apply_smoke(args: argparse.Namespace) -> None:
    """把规模缩到 CPU 上几秒钟跑得完。**数字没有任何意义**，只证明管线通。

    单独成一个函数是为了能被测试直接盯着——冒烟规模一旦悄悄变大，CI 会先受不了，
    而那时报出来的是"超时"，看不出真正的原因。
    """
    args.seeds = "0"
    args.epochs = 1
    args.n_rec = 32
    args.n_train = 256
    args.out_dir = Path("data/_smoke_bptt")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        apply_smoke(args)
        print("冒烟模式：这里的数字没有任何意义，只证明管线能跑通。")

    seeds = parse_seeds(args.seeds)
    device = select_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"BPTT 臂：{len(seeds)} 个种子（{seeds[0]}–{seeds[-1]}），数据集 {args.dataset}，"
        f"eta={args.learning_rate}，epochs={args.epochs}，设备 {device}"
    )

    records = []
    for seed in seeds:
        print(f"\n{'=' * 78}\n种子 {seed}\n{'=' * 78}", flush=True)
        record = run_seed(args, seed, device)
        records.append(record)
        out_path = args.out_dir / f"seed{seed}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[落盘] {out_path}", flush=True)

    accuracies = [r["test_accuracy"] for r in records]
    actives = [r["active_fraction"] for r in records]
    print(f"\n{'=' * 78}")
    print(f"BPTT 臂汇总（{len(seeds)} 个种子：{seeds[0]}–{seeds[-1]}，{args.dataset}）")
    print(f"{'=' * 78}")
    for record in records:
        print(
            f"  种子 {record['seed']:>3}  准确率 {record['test_accuracy']:.4f}  "
            f"活跃比例 {record['active_fraction']:.4f}  耗时 {record['elapsed_s']:.0f} s"
        )
    mean = statistics.fmean(accuracies)
    print(
        f"\n  测试准确率        均值 {mean:.4f}  最小 {min(accuracies):.4f}  "
        f"最大 {max(accuracies):.4f}  极差 {(max(accuracies) - min(accuracies)) * 100:.2f} 个百分点"
    )
    print(f"  活跃神经元比例      均值 {statistics.fmean(actives):.4f}")
    print(f"  JSON 落在 {args.out_dir}/，逐种子可查")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
