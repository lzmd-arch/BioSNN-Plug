"""BPTT 对照基线：同一架构用代理梯度训练，产出量化差距报告。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

计划书 §九 把「e-prop 序列任务 PPL／可用；**产出与代理梯度基线的量化差距报告**」列在
**第二阶段**，§七 第一阶段只要求"顺序任务可用"。所以本文件是**前期铺垫**，不是第一
阶段的验收项——它存在的意义是让"e-prop 比 BPTT 差多少"这个数字在需要时随手可得，而
不是等第二阶段再从头搭对照。

## 为什么这个对照是有意义的

两者用**完全相同的**网络（:class:`~research.eprop.neurons.ALIFCell`）、相同的损失、
相同的优化预算，唯一差别是：

* e-prop：局部资格痕迹 + 每神经元一个标量的学习信号，**无跨层梯度**；
* BPTT：把整段时间展开求梯度，用**伪导数**（surrogate gradient）穿过脉冲。

脉冲函数的反向本来就由 :func:`~research.eprop.neurons.spike_function` 用伪导数定义，
所以 BPTT 这条路径不需要额外实现——只是不再切断计算图。

## 差距该被怎么读

这个数字**不是**"e-prop 不如 BPTT"的判决，而是那条近似（丢掉跨神经元循环路径与复位
项带来的脉冲介导路径，见 ``traces.py`` 的说明）在**这个任务、这个规模**上的代价。
换任务与规模，差距会变。所以报告里必须带上配置。
"""

from __future__ import annotations

import argparse
import time

import torch

from research.common.device import measure_peak_memory
from research.common.provenance import collect
from research.eprop.neurons import ALIFCell, ALIFState
from research.eprop.traces import exp_convolve

__all__ = ["BPTTLearner", "compare"]


class BPTTLearner(torch.nn.Module):
    """同一架构，但用 BPTT + 代理梯度训练。

    与 :class:`~research.eprop.eprop.EPropLearner` 的唯一结构性差别是：前向**保留计算图**，
    用 ``loss.backward()`` 更新，而不是用资格痕迹。
    """

    def __init__(
        self,
        n_in: int,
        n_rec: int,
        n_out: int,
        *,
        tau: float = 20.0,
        tau_adaptation: float = 500.0,
        threshold: float = 0.62,
        beta: float = 0.07,
        dampening_factor: float = 0.3,
        n_refractory: int = 2,
        learning_rate: float = 1e-3,
        decay_out: float = 0.95,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.cell = ALIFCell(
            n_in,
            n_rec,
            tau=tau,
            tau_adaptation=tau_adaptation,
            threshold=threshold,
            beta=beta,
            dampening_factor=dampening_factor,
            n_refractory=n_refractory,
            generator=generator,
        )
        out_generator = generator or torch.Generator()
        self.w_out = torch.nn.Parameter(
            torch.randn(n_rec, n_out, generator=out_generator) / (n_rec**0.5)
        )
        self.decay_out = float(decay_out)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """返回最后一个时间步的 logits，``(batch, n_out)``。**保留计算图。**"""
        steps, batch = inputs.shape[0], inputs.shape[1]
        state = ALIFState.zeros(batch, self.cell.n_rec, device=inputs.device, dtype=inputs.dtype)
        spikes = []
        for t in range(steps):
            z, state = self.cell.step(inputs[t], state)
            spikes.append(z)
        filtered = exp_convolve(torch.stack(spikes), self.decay_out)
        return filtered[-1] @ self.w_out

    def update(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """一步 BPTT 更新，返回更新前的损失。"""
        self.optimizer.zero_grad()
        logits = self.forward(inputs)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        value = float(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return value

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs).argmax(dim=-1)


def compare(args: argparse.Namespace) -> dict[str, float]:
    """用同一套数据与预算训练 e-prop 与 BPTT，返回两者的测试准确率。"""
    from research.eprop.eprop import EPropLearner
    from research.eprop.tasks import load_smnist

    train_set = load_smnist(args.data_dir, split="train", seed=args.seed)
    test_set = load_smnist(args.data_dir, split="test", seed=args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(train_set), generator=generator)[: args.n_train]
    train_inputs, train_labels = train_set.batch(order)
    test_inputs, test_labels = test_set.batch(torch.arange(len(test_set)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

    common = {
        "tau_adaptation": args.tau_adaptation,
        "threshold": args.threshold,
        "beta": args.beta,
        "generator": torch.Generator().manual_seed(args.seed),
    }
    results: dict[str, float] = {}

    for name, model in (
        (
            "e-prop",
            EPropLearner(
                train_inputs.shape[-1],
                args.n_rec,
                10,
                learning_rate_in=args.learning_rate,
                learning_rate_rec=args.learning_rate,
                learning_rate_out=args.learning_rate_out,
                decay_out=args.decay_out,
                **common,
            ),
        ),
        (
            "BPTT",
            BPTTLearner(
                train_inputs.shape[-1],
                args.n_rec,
                10,
                learning_rate=args.learning_rate_bptt,
                decay_out=args.decay_out,
                **common,
            ),
        ),
    ):
        model = model.to(device)
        started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            shuffle = torch.randperm(
                len(train_labels), generator=torch.Generator().manual_seed(epoch)
            )
            running = 0.0
            batches = 0
            for start in range(0, len(shuffle), args.batch_size):
                chunk = shuffle[start : start + args.batch_size]
                running += model.update(train_inputs[:, chunk], train_labels[chunk])
                batches += 1
            print(
                f"[{name}] epoch {epoch:3d}  train_ce {running / max(batches, 1):.4f}  "
                f"({time.perf_counter() - started:.0f}s)",
                flush=True,
            )
        accuracy = float((model.predict(test_inputs) == test_labels).float().mean().item())
        results[name] = accuracy
        print(f"[{name}] 测试准确率 {accuracy:.4f}", flush=True)

    results["gap"] = results["BPTT"] - results["e-prop"]
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-rec", type=int, default=256)
    parser.add_argument(
        "--n-train", type=int, default=20000, help="训练样本数（对照要跑两遍，取小一点）"
    )
    parser.add_argument("--tau-adaptation", type=float, default=500.0)
    parser.add_argument("--threshold", type=float, default=0.62)
    parser.add_argument("--beta", type=float, default=0.07)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--learning-rate-out", type=float, default=5e-2)
    parser.add_argument("--learning-rate-bptt", type=float, default=5e-3)
    parser.add_argument("--decay-out", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    with measure_peak_memory(device) as memory:
        results = compare(args)
    elapsed = time.perf_counter() - started
    print(
        f"\n量化差距报告：e-prop {results['e-prop']:.4f} vs BPTT {results['BPTT']:.4f}"
        f"（差距 {results['gap']:+.4f}）"
    )
    print(
        f"配置：n_rec={args.n_rec}, epochs={args.epochs}, n_train={args.n_train}, "
        f"batch={args.batch_size}, seed={args.seed}\n"
        f"**这个数字只在上述配置下成立**——它衡量的是 e-prop 那条近似在此任务此规模上的"
        f"代价，换配置会变。"
    )

    record = collect(
        "eprop/bptt-gap",
        # **这一格此前是空的**——`collect` 的 `seeds` 是可选参数，漏传不会报错，只会让复现记录
        # 块里多出一行没有值的「随机种子：」。种子的实际值当时只写在备注行里，信息没丢但不整齐；
        # 而「哪个字段该有值」正是这种自证文件存在的意义，空着比写错更糟——它看起来像没控制种子。
        seeds=f"seed={args.seed}（对照的两条臂共用：同一份数据顺序 + 同一套初始权重）",
        elapsed_s=elapsed,
        peak_mb=memory["peak_mb"],
        device=device,
        notes=[
            f"两条线同架构、同损失、同预算，唯一差别是学习规则："
            f"n_rec={args.n_rec}, epochs={args.epochs}, n_train={args.n_train}, "
            f"batch={args.batch_size}, seed={args.seed}",
            f"e-prop: eta_in=eta_rec={args.learning_rate}, eta_out={args.learning_rate_out}；"
            f"BPTT: eta={args.learning_rate_bptt}，代理梯度穿过脉冲函数",
            "配对设计：两者用同一个 seed（同一份数据顺序、同一套初始权重）",
            "**这个差距不是「e-prop 不如 BPTT」的判决**——它是那条近似（丢掉跨神经元循环路径与"
            "复位项带来的脉冲介导路径，见 traces.py 的说明）在此任务此规模上的代价",
        ],
    )
    print("\n" + record.render_block())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
