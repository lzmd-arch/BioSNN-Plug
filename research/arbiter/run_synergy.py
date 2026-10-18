"""协同性基准的 runner：H / E / MIX 三臂在**同一个共享输入权重**上。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

规格：[`sweep_results/arbiter_synergy_task_spec.md`](../../sweep_results/arbiter_synergy_task_spec.md)。

## 公平比较的做法

三条臂的差别**只在共享权重 W（28 → 256）怎么被更新**。为了不让"换了个读出"混淆结果，
全部臂都用同一套东西：

* 特征：``relu(x @ W)``（对每步的 28 维输入）；
* 读出：同一个 delta 线性读出，只用**有标签子集**训练；
* 评测：同一份测试集。

## 三条臂

| 臂 | W 的更新 |
| :--- | :--- |
| H | 只用 Hebbian 的 ΔW（`alpha=0`） |
| E | 只用 e-prop 的 ΔW（`alpha=1`） |
| MIX | 仲裁器按 `alpha` 混合两者 |

⚠️ **ΔW 的取法**：两条规则都在自己的 `update()` 里直接改权重，所以这里用「快照 → 跑一步 →
差值 → 还原」，再交给仲裁器写回。这比改两条线的内部实现安全——**不改已验证的代码路径**。

⚠️ **训练量必须够**。实测（种子 0、批量 64、初始化附近）：Hebbian 每批移动约 `2.8e-5`、
e-prop 约 `2e-4`，而共享权重的范数是 16。**1 个 epoch 累计变化约 1e-2，三臂因此给出
完全相同的准确率**（实测都是 0.254）。这不是接线错误：两条规则在各自的线里都要跑几十个
epoch 才成形。跑基准前先按 1 个 epoch 量耗时，再把 epoch 数定到让 W 真正被塑造的量级。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from research.arbiter.es_arbiter import ESArbiter
from research.arbiter.synergy_task import label_mask, make_permutation, permute_inputs
from research.common.provenance import collect
from research.continual.readout import DeltaReadout, evaluate  # 复用同一读出
from research.eprop.tasks import load_smnist

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "arbiter_synergy"

__all__ = ["Arm", "features", "run_arm"]


#: 计划书 §七 第二阶段的判据在这里换算成比例：随机基线 = 1/10。
CHANCE = 0.1


@dataclass(frozen=True)
class Arm:
    """一条实验臂。``kind`` 决定 W 怎么更新。"""

    name: str
    kind: str  # "hebbian" | "eprop" | "mix"
    alpha: float | None = None  # mix 时由 ESArbiter 学；否则固定


def features(inputs: torch.Tensor, weight: torch.Tensor) -> np.ndarray:
    """``(T, n, 28)`` 序列 → ``(n, 256)`` 的 ``relu(x @ W)`` 时间平均。"""
    # 每条样本在 T 个时间步上的平均输入，再乘共享权重
    averaged = inputs.mean(dim=0)  # (n, 28)
    return torch.relu(averaged @ weight).detach().cpu().numpy().astype(np.float64)


def _hebbian_update(hebbian, inputs, labels) -> torch.Tensor:
    """跑一步 Hebbian 的局部规则，返回它对共享权重的 ΔW（形状与 W 同）。

    ``Inputs`` 是 ``(T, n, 28)``；Hebbian 用时间平均作为静态输入。
    """
    layer = hebbian.layers[0]
    before = layer.linear.weight.detach().clone()
    averaged = inputs.mean(dim=0)
    kernel = hebbian.label_kernel(labels) if labels is not None else None
    layer(averaged, kernel, update=True)
    delta = layer.linear.weight.detach() - before
    layer.linear.weight.data.copy_(before)  # 还原：混合由调用方负责
    return delta  # (n_shared, n_in)，与 ib_hebbian 的存储约定一致


def _eprop_update(eprop, inputs, labels) -> torch.Tensor:
    """跑一步 e-prop，返回它对共享输入权重的 ΔW（``(n_in, n_shared)``）。"""
    before = eprop.cell.w_in.detach().clone()
    eprop.update(inputs, labels)
    delta = eprop.cell.w_in.detach() - before
    eprop.cell.w_in.data.copy_(before)
    return delta


def run_arm(
    arm: Arm,
    *,
    seed: int,
    fraction: float,
    epochs: int,
    batch: int,
    n_shared: int,
    data_dir: Path,
    device: torch.device,
    arbiter: ESArbiter | None = None,
) -> dict:
    from research.arbiter.shared_synapse import SharedSynapseConfig, SharedSynapseHarness

    torch.manual_seed(seed)
    np.random.seed(seed)

    train = load_smnist(data_dir, split="train", seed=seed)
    test = load_smnist(data_dir, split="test", seed=seed)
    perm = make_permutation(28, seed=seed)
    train_inputs = permute_inputs(train.inputs, perm).to(device)
    test_inputs = permute_inputs(test.inputs, perm).to(device)
    labels = train.labels
    mask = label_mask(labels, fraction, seed=seed)

    config = SharedSynapseConfig(n_in=28, n_shared=n_shared, n_classes=10)
    harness = SharedSynapseHarness(
        config,
        arbiter=arbiter or ESArbiter(alpha=0.0 if arm.kind == "hebbian" else 1.0, seed=seed),
        seed=seed,
    )
    # harness 在 CPU 上构造；要用 GPU 就把两个模型与共享权重一起搬过去。
    if device.type != "cpu":
        harness.eprop.to(device)
        harness.hebbian.to(device)
        harness._shared = harness._shared.to(device)
        harness.sync_from_shared()
    weight = harness.shared_weight

    readout = DeltaReadout(n_shared, 10, learning_rate=0.01)
    n = train_inputs.shape[1]
    for _ in range(epochs):
        order = torch.randperm(n)
        for start in range(0, n, batch):
            idx = order[start : start + batch]
            x = train_inputs[:, idx]
            y = labels[idx]

            if arm.kind in ("hebbian", "mix"):
                delta_h = _hebbian_update(harness.hebbian, x, y)
            else:
                delta_h = torch.zeros(weight.shape[1], weight.shape[0], device=device)
            if arm.kind in ("eprop", "mix"):
                labeled = mask[idx.cpu().numpy()]
                if labeled.any():
                    delta_e = _eprop_update(
                        harness.eprop, x[:, torch.as_tensor(labeled, device=device)], y[labeled]
                    )
                else:
                    delta_e = torch.zeros_like(weight)
            else:
                delta_e = torch.zeros_like(weight)

            if arm.kind == "mix":
                # blend 的 a=eprop、b=hebbian（与 harness.step 的约定一致）；
                # delta_h 是 (n_shared, n_in)，先转置到 (n_in, n_shared) 再混合。
                blended = harness.arbiter.blend(
                    delta_e.detach().cpu().numpy(), delta_h.detach().cpu().numpy().T
                )
                new_weight = weight.detach().cpu() + torch.as_tensor(blended, dtype=weight.dtype)
            elif arm.kind == "eprop":
                new_weight = weight.detach().cpu() + delta_e.detach().cpu()
            else:
                new_weight = weight.detach().cpu() + delta_h.detach().cpu().T
            harness._shared = new_weight.to(weight.device)
            harness.sync_from_shared()

            # 读出只在有标签的样本上更新
            labeled_all = mask[idx.cpu().numpy()]
            if labeled_all.any():
                feats = features(
                    x[:, torch.as_tensor(labeled_all, device=device)], harness.shared_weight
                )
                readout.step(feats, y[labeled_all].cpu().numpy())

    test_features = features(test_inputs, harness.shared_weight)
    accuracy = evaluate(readout, test_features, test.labels.cpu().numpy(), np.arange(10))
    return {
        "arm": arm.name,
        "kind": arm.kind,
        "fraction": fraction,
        "seed": seed,
        "accuracy": accuracy,
        "alpha": float(harness.arbiter.alpha),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="协同性基准：H / E / MIX")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--fractions", default="0.01")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--shared", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--arms", default="H,E,MIX", help="逗号分隔，可选 H / E / MIX")
    parser.add_argument("--smoke", action="store_true", help="1 种子、1% 标签、小批量")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    if args.smoke:
        args.epochs = 1
        args.batch = 64

    catalogue = {
        "H": Arm("H", "hebbian"),
        "E": Arm("E", "eprop"),
        "MIX": Arm("MIX", "mix"),
    }
    wanted = [name.strip() for name in args.arms.split(",") if name.strip()]
    unknown = [name for name in wanted if name not in catalogue]
    if unknown:
        raise SystemExit(f"未知的臂：{unknown}；可选 {sorted(catalogue)}")
    arms = [catalogue[name] for name in wanted]
    results = []
    started = time.perf_counter()
    for fraction in (float(f) for f in args.fractions.split(",")):
        for arm in arms:
            for seed in (int(s) for s in args.seeds.split(",")):
                row = run_arm(
                    arm,
                    seed=seed,
                    fraction=fraction,
                    epochs=args.epochs,
                    batch=args.batch,
                    n_shared=args.shared,
                    data_dir=args.data_dir,
                    device=torch.device(args.device),
                )
                results.append(row)
                print(
                    f"  fraction={fraction:.2f} arm={arm.name:3s} seed={seed} "
                    f"acc={row['accuracy']:.3f} alpha={row['alpha']:.2f}"
                )

    summary = {
        "experiment": "arbiter synergy benchmark (sMNIST, fixed permutation, few labels)",
        "chance": CHANCE,
        "config": {
            "epochs": args.epochs,
            "batch": args.batch,
            "n_shared": args.shared,
        },
        "rows": results,
        "repro": collect(
            "仲裁器协同性基准（P0）",
            seeds=args.seeds,
            elapsed_s=time.perf_counter() - started,
            notes=["固定随机置换；分层少量标签；三臂共用 relu(x@W) 特征与同一 delta 读出"],
            device=torch.device(args.device),
        ).render_block(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
