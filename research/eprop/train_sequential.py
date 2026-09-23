"""W2 验收运行脚本：e-prop 在顺序任务上。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

用法::

    uv run python scripts/download_data.py mnist              # 先备数据
    uv run python -m research.eprop.train_sequential          # 验收运行（sMNIST）
    uv run python -m research.eprop.train_sequential --smoke  # CI 用的缩小规模冒烟

    uv run python scripts/download_data.py shd                # 旁证数据集（约 169 MB）
    uv run python -m research.eprop.train_sequential --dataset shd --device cuda

验收标准（计划书 §七 第一阶段）：**顺序任务可用，活跃神经元比例 > 60%**。

**类别数从数据里读**（sMNIST 10 类、SHD 20 类），不写死——写死 10 在 SHD 上会让
``cross_entropy`` 的 target 越界。

计划书没给"可用"的数字，所以本脚本把它拆成三件可查的事：

1. 测试准确率**显著高于随机**（基线取 1/类别数）；
2. **活跃神经元比例 > 60%**——§七 的硬指标，也是 §3.1「死亡神经元防护」的触发线；
3. 产出与代理梯度 BPTT 基线的**量化差距报告**（见 ``bptt_baseline.py``）。

第 3 项 §九 列在第二阶段，这里作为前期铺垫产出，**不作为第一阶段的验收项**。

## 编码的选择会写进结论边界

MNIST 的模拟像素按**伯努利采样**变成脉冲，SHD 的脉冲时刻按固定时间窗铺进 100 个箱
（见 :mod:`research.eprop.tasks`）。两者都是本项目自己的编码选择，不是任何论文的规定。
换编码结果会变，所以种子固定并写进复现记录。
"""

from __future__ import annotations

import argparse
import time

import torch

from research.common.device import describe_device, measure_peak_memory, select_device
from research.common.provenance import DegradationLog, collect
from research.common.seeding import SeedBook
from research.eprop.eprop import EPropLearner
from research.eprop.tasks import load_shd, load_smnist

#: 计划书 §3.1 与 §七 的阈值。
ACTIVE_NEURON_THRESHOLD = 0.60


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dataset", default="smnist", choices=["smnist", "shd"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-rec", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.07, help="ALIF 适应强度")
    parser.add_argument("--threshold", type=float, default=0.62)
    parser.add_argument("--tau-adaptation", type=float, default=500.0)
    parser.add_argument("--learning-rate-in", type=float, default=2e-3)
    parser.add_argument("--learning-rate-rec", type=float, default=2e-3)
    parser.add_argument("--learning-rate-out", type=float, default=5e-2)
    parser.add_argument("--decay-out", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--smoke", action="store_true", help="CI 冒烟：缩到 CPU 一分钟内")
    return parser


def apply_smoke_overrides(args: argparse.Namespace) -> int:
    args.epochs = 2
    args.batch_size = 32
    args.n_rec = 32
    args.device = "cpu"
    return 512


def load_task(args: argparse.Namespace, seed: int):
    """返回 ``(train, test, n_in, n_classes)``。

    **类别数从数据里读，不写死。** sMNIST 是 10 类、SHD 是 20 类；写死 10 在 SHD 上会让
    ``cross_entropy`` 的 target 越界（CUDA 上只报 device-side assert，看不出真正的原因）。
    这条路径在第一次真跑之前一直是写死的。
    """
    loader = load_shd if args.dataset == "shd" else load_smnist
    train = loader(args.data_dir, split="train", seed=seed)
    test = loader(args.data_dir, split="test", seed=seed)
    n_classes = int(max(int(train.labels.max()), int(test.labels.max()))) + 1
    return train, test, train.inputs.shape[-1], n_classes


def train(args: argparse.Namespace, device: torch.device) -> float:
    book = SeedBook(base=args.seed)
    book.apply("全局种子")

    limit = apply_smoke_overrides(args) if args.smoke else None
    train_set, test_set, n_in, n_classes = load_task(args, book.derive("脉冲编码"))

    n_samples = min(len(train_set), limit) if limit else len(train_set)
    n_val = max(n_samples // 10, 1)
    generator = torch.Generator().manual_seed(book.derive("划分验证集"))
    order = torch.randperm(len(train_set), generator=generator)[:n_samples]
    val_index, train_index = order[:n_val], order[n_val:]

    train_inputs, train_labels = train_set.batch(train_index)
    val_inputs, val_labels = train_set.batch(val_index)
    test_inputs, test_labels = test_set.batch(torch.arange(len(test_set)))
    if limit:
        test_inputs = test_inputs[:, : max(limit // 4, 32)]
        test_labels = test_labels[: max(limit // 4, 32)]

    train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

    model = EPropLearner(
        n_in,
        args.n_rec,
        n_classes,
        tau_adaptation=args.tau_adaptation,
        threshold=args.threshold,
        beta=args.beta,
        learning_rate_in=args.learning_rate_in,
        learning_rate_rec=args.learning_rate_rec,
        learning_rate_out=args.learning_rate_out,
        decay_out=args.decay_out,
        generator=torch.Generator().manual_seed(book.derive("权重初始化")),
    ).to(device)

    shuffle = torch.Generator().manual_seed(book.derive("每批打乱"))
    started = time.perf_counter()
    n_train = len(train_labels)

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

            if epoch == 1 or epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
                val_accuracy = float(
                    (model.predict(val_inputs) == val_labels).float().mean().item()
                )
                print(
                    f"epoch {epoch:3d}  train_ce {running:.4f}  val_acc {val_accuracy:.4f}",
                    flush=True,
                )

    elapsed = time.perf_counter() - started

    test_prediction = model.predict(test_inputs)
    test_accuracy = float((test_prediction == test_labels).float().mean().item())
    active = model.active_neuron_fraction(test_inputs)
    sparsity = model.spike_sparsity(test_inputs)

    print(f"\n测试准确率：{test_accuracy:.4f}（{n_classes} 类随机基线 {1 / n_classes:.2f}）")
    print(f"活跃神经元比例：{active:.4f}（§七 阈值 > {ACTIVE_NEURON_THRESHOLD}）")
    print(f"脉冲稀疏度：{sparsity:.4f}")

    degradation = DegradationLog()
    if limit is not None:
        degradation.scale_reduction = True
        degradation.notes.append(f"冒烟模式：只用 {n_samples} 个训练样本、{args.epochs} epoch")

    record = collect(
        f"eprop/{args.dataset}",
        seeds=book.render(),
        elapsed_s=elapsed,
        gpu=describe_device(device).render(),
        peak_mb=memory["peak_mb"],
        degradation=degradation,
        notes=[
            f"n_rec={args.n_rec}, beta={args.beta}, thr={args.threshold}, "
            f"eta_in={args.learning_rate_in}, eta_rec={args.learning_rate_rec}, "
            f"eta_out={args.learning_rate_out}, decay_out={args.decay_out}",
            "输入编码："
            + (
                "像素值作发放概率的伯努利采样（本项目自己的选择）"
                if args.dataset == "smnist"
                else "SHD 脉冲时刻铺到 100 个箱、箱内二值化（本项目自己的选择）"
            ),
        ],
    )
    print("\n" + record.render_block())

    return test_accuracy, active, n_classes


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = select_device(args.device)
    test_accuracy, active, n_classes = train(args, device)

    if args.smoke:
        return 0

    chance = 1.0 / n_classes
    passed_activity = active > ACTIVE_NEURON_THRESHOLD
    passed_accuracy = test_accuracy > 2 * chance  # 显著高于随机
    print(
        f"\n验收（计划书 §七）：活跃神经元比例 {active:.4f} > "
        f"{ACTIVE_NEURON_THRESHOLD} → {'达到' if passed_activity else '未达到'}；"
        f"准确率 {test_accuracy:.4f} > {2 * chance:.2f}（{n_classes} 类的两倍随机基线）"
        f" → {'达到' if passed_accuracy else '未达到'}。"
    )
    return 0 if (passed_activity and passed_accuracy) else 1


if __name__ == "__main__":
    raise SystemExit(main())
