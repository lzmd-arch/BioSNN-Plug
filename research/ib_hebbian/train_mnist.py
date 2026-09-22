"""W1 验收运行脚本：核化 IB-Hebbian 感知层在 MNIST 上。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

用法::

    uv run python scripts/download_data.py mnist               # 先备数据
    uv run python -m research.ib_hebbian.train_mnist           # 验收运行（本机 GPU）
    uv run python -m research.ib_hebbian.train_mnist --smoke   # CI 用的缩小规模冒烟

验收标准（计划书 §七 第一阶段）：**MNIST 70%+**。

论文的对照数字：3 层 × 1024 全连接、高斯核 + 分组 + 除法归一化，MNIST 上 5 次运行
平均 **98.1%**（backprop 98.6%）。所以 70% 这个门槛余量很大——若实测远低于它，
说明实现有问题，而不是任务太难。

## 超参来源（免得以后有人问"这些数哪来的"）

| 参数 | 值 | 出处 |
| :--- | :--- | :--- |
| 层数 / 宽度 | 3 × 1024 | §5.2 |
| 分组数 c_k | 16 或 32 | Table 3（MNIST 行给出 {16, 32}） |
| 高斯核 σ | 5 | D.5 |
| 瓶颈平衡 γ | 2 | D.5（§5.2 说别的取值更差） |
| 除法归一化 p | 0.2 | D.5（backprop / pHSIC） |
| 平滑偏移 δ | 1 | D.5 |
| 分组指数 q | 1 − p = 0.8 | 官方 argparse：'should be 1 - divnorm_power for Hebbian updates' |
| 非线性 | LReLU，负斜率 0.01 | 官方 argparse 默认值。论文 §5.1 写 0.1、D.5 写 0.01，两者冲突，以代码为准 |
| 动量 | 0.95 | D.5（SGD） |
| 局部损失权重衰减 | 1e-7 | D.5 |
| dropout | 0.01 | D.6 |
| 批大小 | 256 | §5.2 |
| epoch | 100 | D.6 |
| 学习率衰减 | ×0.25 @ 50/75/90 epoch | D.6 |
| 数据预处理 | (x/255 − 0.5)/0.5 | D.4 |
| 验证集 | 训练集的 10% | §5.1 |

**η_l（局部学习率）没有照抄。** Table 3 的列与 PDF 抽取出的数值对不齐（10 列数值、
只有 6 个 η_l 与 6 个 c_k），逐列映射无法可靠还原。论文自己也是在验证集上调的
（D.5），所以这里按同样做法处理，并在复现记录里写下实际取值。编一个精确数字写进
注释要比这更不诚实。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from research.common.device import describe_device, measure_peak_memory, select_device
from research.common.metrics import active_neuron_fraction, spike_sparsity
from research.common.provenance import DegradationLog, collect
from research.common.seeding import SeedBook
from research.ib_hebbian.model import IBHebbianPerceptron, preprocess_mnist
from research.ib_hebbian.readout import RidgeReadout

#: 学习率衰减点，写成总 epoch 的比例。论文 D.6 的 50/75/90 是对 100 epoch 而言，
#: 正好等于 0.5/0.75/0.9，所以取 100 epoch 时与原设置逐点一致。
SCHEDULE_FRACTIONS = (0.5, 0.75, 0.9)


def load_mnist(data_dir: str | Path) -> tuple[torch.Tensor, ...]:
    """读取 ``scripts/download_data.py`` 落下的 ``.npy``。"""
    root = Path(data_dir) / "mnist"
    required = {
        "train_images": root / "train_images.npy",
        "train_labels": root / "train_labels.npy",
        "test_images": root / "test_images.npy",
        "test_labels": root / "test_labels.npy",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise SystemExit(
            f"缺少 {root} 下的数据文件：{missing}\n"
            f"请先运行：uv run python scripts/download_data.py mnist"
        )
    arrays = {name: np.load(path) for name, path in required.items()}
    return (
        preprocess_mnist(torch.from_numpy(arrays["train_images"])),
        torch.from_numpy(arrays["train_labels"]).long(),
        preprocess_mnist(torch.from_numpy(arrays["test_images"])),
        torch.from_numpy(arrays["test_labels"]).long(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data", help="数据目录（download_data.py 的产物）")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument(
        "--n-groups", type=int, default=32, help="分组数 c_k（Table 3 给 16 或 32）"
    )
    parser.add_argument("--sigma", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--divnorm-power", type=float, default=0.2)
    parser.add_argument("--smoothing-delta", type=float, default=1.0)
    parser.add_argument("--hidden-lr", type=float, default=1.0, help="η_l，局部学习率")
    parser.add_argument(
        "--readout-lr", type=float, default=5e-3, help="η_f，读出学习率（仅 --readout-kind sgd）"
    )
    parser.add_argument(
        "--readout-kind",
        default="ridge",
        choices=["ridge", "sgd"],
        help="读出层怎么学。**默认 ridge**：累积 XᵀX 与 XᵀY 后一次解出岭回归闭式解"
        "（无 backward、无学习率、无迭代）。'sgd' 是原来的交叉熵 + SGD 路径，保留作对照。"
        "**两者都不是「局部」的**——闭式解去掉了反向传播与迭代，但 XᵀX 仍是全局二阶统计量",
    )
    parser.add_argument(
        "--readout-ridge",
        type=float,
        default=None,
        help="岭回归的正则系数 λ（仅 --readout-kind ridge）。**不给就在验证集上选**"
        "（候选 {0, 1e-4, 1e-3, 1e-2, 1e-1}）；给了就用它、跳过选择。偏置项不正则化",
    )
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="CI 冒烟：缩到能在 CPU 上一分钟内跑完；产出不作为验收数字",
    )
    return parser


def apply_smoke_overrides(args: argparse.Namespace) -> int:
    """把参数改成冒烟规模，返回训练样本上限。"""
    args.epochs = 2
    args.batch_size = 64
    args.width = 64
    args.n_groups = 8
    args.n_layers = 2
    args.device = "cpu"
    return 512


def build_ridge_readout(args: argparse.Namespace, device: torch.device) -> RidgeReadout:
    """按 ``--readout-kind ridge`` 造累积器。

    **λ 不在这里给**：``RidgeReadout`` 构造时的 λ 只是 ``solve()`` 不带参数时的兜底，
    而 ``train()`` 里每一次 ``solve()`` 都显式带上候选值（``--readout-ridge`` 或验证集候选集）。
    把 ``args.readout_ridge`` 直接传进来是错的——它的默认值是 ``None``，会让
    「不给 λ 就在验证集上选」这条默认路径当场崩掉（CI 冒烟步骤抓到过）。
    """
    return RidgeReadout(args.width, 10, device=device)


def build_model(args: argparse.Namespace) -> IBHebbianPerceptron:
    return IBHebbianPerceptron(
        28 * 28,
        width=args.width,
        n_layers=args.n_layers,
        n_classes=10,
        n_groups=args.n_groups,
        sigma=args.sigma,
        gamma=args.gamma,
        divnorm_power=args.divnorm_power,
        smoothing_delta=args.smoothing_delta,
        hidden_learning_rate=args.hidden_lr,
        readout_learning_rate=args.readout_lr,
        dropout_p=args.dropout,
    )


def decay_learning_rates(model: IBHebbianPerceptron, factor: float = 0.25) -> None:
    optimizers = [layer.optimizer for layer in model.layers] + [model.readout_optimizer]
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] *= factor


def schedule_epochs(epochs: int) -> set[int]:
    """衰减发生在第几个 epoch（1-based）。100 epoch 时得到 {50, 75, 90}，与 D.6 一致。"""
    return {max(1, round(epochs * fraction)) for fraction in SCHEDULE_FRACTIONS}


def train(args: argparse.Namespace, device: torch.device):
    book = SeedBook(base=args.seed)
    book.apply("全局种子")

    limit = apply_smoke_overrides(args) if args.smoke else None
    train_x_all, train_y_all, test_x, test_y = load_mnist(args.data_dir)

    # 论文 §5.1：验证集取训练集的 10%。
    n_val = len(train_x_all) // 10
    split_generator = torch.Generator().manual_seed(book.derive("划分验证集"))
    permutation = torch.randperm(len(train_x_all), generator=split_generator)
    val_index, train_index = permutation[:n_val], permutation[n_val:]
    if limit is not None:
        train_index, val_index = train_index[:limit], val_index[: max(limit // 8, 32)]

    train_x, train_y = train_x_all[train_index].to(device), train_y_all[train_index].to(device)
    val_x, val_y = train_x_all[val_index].to(device), train_y_all[val_index].to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)

    torch.manual_seed(book.derive("权重初始化"))
    model = build_model(args).to(device)
    decays = schedule_epochs(args.epochs)

    shuffle_generator = torch.Generator().manual_seed(book.derive("每轮打乱"))
    started = time.perf_counter()

    with measure_peak_memory(device) as memory:
        for epoch in range(1, args.epochs + 1):
            if epoch in decays:
                decay_learning_rates(model)

            order = torch.randperm(len(train_x), generator=shuffle_generator)
            running = 0.0
            for start in range(0, len(order), args.batch_size):
                batch = order[start : start + args.batch_size]
                running += model.step(
                    train_x[batch], train_y[batch], update_readout=args.readout_kind == "sgd"
                )
            running /= max(1, len(order) // args.batch_size)

            if epoch == 1 or epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
                if args.readout_kind == "ridge":
                    # **不能打印读出的 CE/acc**：ridge 模式下读出要到训练结束才解出来，
                    # 现在它是随机初始权重，那两列数会随隐藏层特征漂移而乱走，看着像发散。
                    # 打印真正在被优化的量——各隐藏层的局部目标之和。
                    objective = model.hidden_objective(val_x, val_y)
                    print(
                        f"epoch {epoch:3d}  隐藏层局部目标（验证集）{objective:.4f}",
                        flush=True,
                    )
                else:
                    val_loss, val_accuracy = model.evaluate(val_x, val_y)
                    print(
                        f"epoch {epoch:3d}  train_ce {running:.4f}  "
                        f"val_ce {val_loss:.4f}  val_acc {val_accuracy:.4f}",
                        flush=True,
                    )

    # **闭式解读出：训练完之后一次解出来。** 不迭代、不反传、无学习率。
    #
    # 放在循环**外面**而不是里面，是因为隐藏层的特征随训练在变——在线累积会把不同 epoch
    # 的特征混在一起，解出来的 W 不对应任何一组特征。标准做法就是冻结特征后解一次。
    #
    # 注意这一步与隐藏层的局部性主张**正交**：隐藏层用的是自己的目标（Eq. 35 的教学信号），
    # 训练期间完全不知道读出是什么。见 readout.py 的边界说明。
    if args.readout_kind == "ridge":
        readout_started = time.perf_counter()
        # 构造时**不给 λ**：这里的 λ 只是 ``solve()`` 不带参数时的兜底，而下面每一次
        # ``solve()`` 都显式带上候选值，兜底用不到。曾经把 ``args.readout_ridge`` 直接传进来——
        # 它的默认值是 None，于是「不给 λ 就在验证集上选」这条**默认路径**一进来就 TypeError，
        # CI 的冒烟步骤抓到的就是这个。造累积器这一步现在收在 ``build_ridge_readout`` 里，
        # 由测试盯着。
        accumulator = build_ridge_readout(args, device)
        with torch.no_grad():
            for start in range(0, len(train_x), args.batch_size):
                stop = start + args.batch_size
                accumulator.accumulate(model.features(train_x[start:stop]), train_y[start:stop])
        # **λ 用验证集选，不拍一个数。** Gram 矩阵只累积一次，多解几个几乎不花时间；
        # 而原来的做法是手写一个 1e-2 就去读测试准确率——那既没有依据，也是在测试集上
        # 选超参。候选集覆盖 0（无正则）到 1e-1，选验证准确率最高的那个。
        # **显式给了 λ 就用它**，否则在验证集上选。原来两个 solve() 调用都传硬编码的候选值，
        # `--readout-ridge` 因此是个从不被读的死参数（审查抓到的）。
        candidates = (
            (args.readout_ridge,)
            if args.readout_ridge is not None
            else (0.0, 1e-4, 1e-3, 1e-2, 1e-1)
        )
        best_ridge, best_accuracy = None, -1.0
        for candidate in candidates:
            weight, bias = accumulator.solve(candidate)
            model.set_readout(weight, bias)
            _, accuracy = model.evaluate(val_x, val_y)
            print(f"  λ={candidate:<7g} 验证准确率 {accuracy:.4f}", flush=True)
            if accuracy > best_accuracy:
                best_ridge, best_accuracy = candidate, accuracy

        # λ 是显式给的还是在验证集上选的，打印与备注里要分得开——否则读记录的人会以为
        # 那个数经过了一轮选择。
        lambda_source = "显式指定" if args.readout_ridge is not None else "由验证集选出"
        model.set_readout(*accumulator.solve(best_ridge))
        print()
        print(
            f"闭式解读出：累积 {accumulator.count} 个训练样本的 (XᵀX, XᵀY)，"
            f"λ = {best_ridge:g}（{lambda_source}，验证准确率 {best_accuracy:.4f}），"
            f"全程耗时 {time.perf_counter() - readout_started:.1f} s",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    test_loss, test_accuracy = model.evaluate(test_x, test_y)

    # 网络健康指标（计划书 §9）。量在归一化之前的活动上——理由见 model.py 的说明。
    activity = model.pre_normalization_activity(test_x)
    active_fraction = active_neuron_fraction(activity)
    sparsity = spike_sparsity(activity)

    print(f"\n测试集：loss {test_loss:.4f}  准确率 {test_accuracy:.4f}")
    print(f"活跃单元比例（归一化前，§9 的速率型类比量）：{active_fraction:.4f}")
    print(f"活动稀疏度：{sparsity:.4f}")

    degradation = DegradationLog()
    if limit is not None:
        degradation.scale_reduction = True
        degradation.notes.append(f"冒烟模式：只用 {limit} 个训练样本、{args.epochs} epoch")

    record = collect(
        "ib_hebbian/mnist",
        seeds=book.render(),
        elapsed_s=elapsed,
        gpu=describe_device(device).render(),
        peak_mb=memory["peak_mb"],
        degradation=degradation,
        notes=[
            f"隐藏层：width={args.width}, n_layers={args.n_layers}, n_groups={args.n_groups}, "
            f"sigma={args.sigma}, gamma={args.gamma}, divnorm_power={args.divnorm_power}, "
            f"eta_local={args.hidden_lr}, dropout={args.dropout}",
            f"读出：kind={args.readout_kind}"
            + (
                f", ridge_lambda={best_ridge:g}（{lambda_source}；累积 XᵀX / XᵀY 后一次解出，无反向传播、无迭代）"
                if args.readout_kind == "ridge"
                else f", eta_readout={args.readout_lr}（交叉熵 + SGD）"
            ),
            "局部性：隐藏层逐层用自身目标就地更新、输入被切断梯度；读出的 (XᵀX, XᵀY) 是全局二阶统计量",
        ],
    )
    print("\n" + record.render_block())

    return test_accuracy


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = select_device(args.device)
    accuracy = train(args, device)

    if not args.smoke:
        threshold = 0.70
        verdict = "达到" if accuracy >= threshold else "未达到"
        print(f"\n验收（计划书 §七）：MNIST 70%+ → {accuracy:.4f}，{verdict}。")
        return 0 if accuracy >= threshold else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
