"""协同性基准的 runner（v2）：H / E / MIX 三臂在**同一个共享输入权重**上。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

规格与设计缺陷记录：[`sweep_results/arbiter_synergy_task_spec.md`](../../sweep_results/arbiter_synergy_task_spec.md)。

## v1 的两个缺陷与 v2 的修法

| v1 的做法 | 后果 | v2 |
| :--- | :--- | :--- |
| 特征 = ``relu(mean_t x @ W)`` | 先把时间维平均掉，对 W 的扰动不敏感；两个不同的 W 给出**逐位相同**的准确率 | 改成 ``mean_t relu(x_t @ W)``：整流在时间步上做 |
| 三臂各自训练在线 delta 读出 | 测到的是"读出 + 权重"的混合；随机权重也能让读出学到 20–35%，H 从不失败 | **共用闭式岭回归读出**，三臂完全相同 |
| 没有检查权重到底动了没有 | 无法区分"规则无效"与"更新没生效" | 加门禁：``‖ΔW‖/‖W₀‖`` 低于 ``GATE_RATIO`` 即整轮作废 |

## 门禁为什么必须有

v1 实测：Hebbian 每批移动约 ``2.8e-5``，而 ``‖W‖ = 16``。若不检查这一点，一条"两臂给出同一个数"
的结果可以被误读成"两臂能力相同"——实际是**更新根本没进到权重里**。

## GPU

本 runner **只在 CPU 上跑**：Hebbian 的 `label_kernel` 在 CPU 上建张量，把模型搬到 GPU 会在
`label_kernel` 里报设备不一致。改它要动已验证的库代码，不值得。CPU 实测约 17 s/臂·epoch。
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
from research.arbiter.separability import fisher_ratio, linear_probe_accuracy
from research.arbiter.synergy_task import label_mask, make_permutation, permute_inputs
from research.common.provenance import collect
from research.eprop.tasks import load_smnist

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "arbiter_synergy"

__all__ = ["Arm", "features", "run_arm"]

#: 随机基线：MNIST 是 10 类。
CHANCE = 0.1

#: 门禁：训练前后 ``‖ΔW‖/‖W₀‖`` 低于这个值 ⇒ 臂的差异没进到权重里，整轮作废。
GATE_RATIO = 0.05

#: 共用闭式读出的 λ。**读出不是本实验的自变量**，固定不调。
RIDGE_LAMBDA = 1.0


@dataclass(frozen=True)
class Arm:
    """一条实验臂。``kind`` 决定共享权重怎么被更新。"""

    name: str
    kind: str  # "hebbian" | "eprop" | "mix"


def features(inputs: torch.Tensor, weight: torch.Tensor) -> np.ndarray:
    """``(T, n, n_in)`` 序列 → ``(n, n_shared)``：**逐时间步整流后再平均**。

    第一版是 ``relu(mean_t(x) @ W)``——先把时间维平均掉再乘 W，结果对 W 的扰动不敏感
    （实测两个不同的 W 给出逐位相同的准确率）。整流放在每个时间步上、平均放在最后，
    W 的扰动因此能改变哪些单元被激活。
    """
    per_step = torch.relu(inputs @ weight)
    return per_step.mean(dim=0).detach().cpu().numpy().astype(np.float64)


def _ridge_accuracy(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[float, int]:
    """闭式岭回归读出，返回 ``(准确率, 有标签样本数)``。

    三臂用**完全相同**的读出——差异只能来自共享权重。
    """
    n_labeled = int(train_features.shape[0])
    if n_labeled == 0:
        return float("nan"), 0
    x = np.asarray(train_features, dtype=np.float64)
    y = np.asarray(train_labels, dtype=int)
    design = np.hstack([x, np.ones((x.shape[0], 1))])
    target = np.zeros((x.shape[0], 10), dtype=np.float64)
    target[np.arange(x.shape[0]), y] = 1.0
    gram = design.T @ design + RIDGE_LAMBDA * np.eye(design.shape[1])
    weights = np.linalg.solve(gram, design.T @ target)

    test_design = np.hstack(
        [np.asarray(test_features, dtype=np.float64), np.ones((test_features.shape[0], 1))]
    )
    predicted = np.argmax(test_design @ weights, axis=1)
    return float(np.mean(predicted == np.asarray(test_labels, dtype=int))), n_labeled


def _ncm_accuracy(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[float, int]:
    """最近类均值（NCM）读出：**没有可拟合的容量**，只看特征的几何。

    为什么要它：闭式岭回归在 256 维、几百个标签上总能到 50% 左右——**W 动不动都是这个数**
    （实测 H 臂从 0 动到 71% 权重变化，岭回归读出始终 0.51）。NCM 只有一个均值向量可估，
    特征若没把类分开就上不去，因此对"表示是否被规则塑造"更敏感。
    """
    n_labeled = int(train_features.shape[0])
    if n_labeled == 0:
        return float("nan"), 0
    x = np.asarray(train_features, dtype=np.float64)
    y = np.asarray(train_labels, dtype=int)
    centroids = np.stack([x[y == c].mean(axis=0) for c in range(10)])
    test = np.asarray(test_features, dtype=np.float64)
    distance = ((test[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    predicted = np.argmin(distance, axis=1)
    return float(np.mean(predicted == np.asarray(test_labels, dtype=int))), n_labeled


def _hebbian_update(hebbian, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """跑一步 Hebbian 的局部规则，返回它对共享权重的 ΔW（``(n_shared, n_in)``）。"""
    layer = hebbian.layers[0]
    before = layer.linear.weight.detach().clone()
    averaged = inputs.mean(dim=0)
    kernel = hebbian.label_kernel(labels) if labels is not None else None
    layer(averaged, kernel, update=True)
    delta = layer.linear.weight.detach() - before
    layer.linear.weight.data.copy_(before)  # 还原：混合由调用方负责
    return delta


def _eprop_update(eprop, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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
    arbiter: ESArbiter | None = None,
    fit_steps: int = 200,
    fit_alpha: bool = True,
    shuffle_labels: bool = False,
    alpha: float = 0.5,
) -> dict:
    from research.arbiter.shared_synapse import SharedSynapseConfig, SharedSynapseHarness

    torch.manual_seed(seed)
    np.random.seed(seed)

    train = load_smnist(data_dir, split="train", seed=seed)
    test = load_smnist(data_dir, split="test", seed=seed)
    perm = make_permutation(28, seed=seed)
    train_inputs = permute_inputs(train.inputs, perm)
    test_inputs = permute_inputs(test.inputs, perm)
    labels = train.labels
    if shuffle_labels:
        # 打乱标签对照：训练信号与样本无关，但评测仍用真实标签（见规格的对照表）。
        labels = labels[
            torch.randperm(labels.shape[0], generator=torch.Generator().manual_seed(seed))
        ]
    mask = label_mask(labels, fraction, seed=seed)

    config = SharedSynapseConfig(n_in=28, n_shared=n_shared, n_classes=10)
    harness = SharedSynapseHarness(
        config,
        # ⚠️ blend(a, b) = (1-α)·a + α·b，而 a=e-prop、b=Hebbian：
        # **α=1 是纯 Hebbian、α=0 是纯 e-prop**。MIX 从 0.5 起步。
        arbiter=arbiter
        or ESArbiter(alpha={"hebbian": 1.0, "eprop": 0.0}.get(arm.kind, alpha), seed=seed),
        seed=seed,
    )
    initial = harness.shared_weight.detach().clone()
    weight = harness.shared_weight

    fit_trace: list[tuple[torch.Tensor, torch.Tensor]] = []
    fitted = False
    n = train_inputs.shape[1]
    for _ in range(epochs):
        order = torch.randperm(n)
        for start in range(0, n, batch):
            idx = order[start : start + batch]
            x = train_inputs[:, idx]
            y = labels[idx]
            labeled = mask[idx.numpy()]

            if arm.kind in ("hebbian", "mix"):
                delta_h = _hebbian_update(harness.hebbian, x, y)
            else:
                delta_h = torch.zeros(weight.shape[1], weight.shape[0])
            if arm.kind in ("eprop", "mix") and labeled.any():
                delta_e = _eprop_update(harness.eprop, x[:, torch.as_tensor(labeled)], y[labeled])
            else:
                delta_e = torch.zeros_like(weight)

            if arm.kind == "mix":
                # 先攒一批更新对，再用 ES 学 α（真实仲裁器路径）；之后用学到的 α 继续。
                if fit_alpha and not fitted:
                    fit_trace.append((delta_e.detach().clone(), delta_h.detach().clone()))
                    if len(fit_trace) >= fit_steps:
                        harness.fit_arbiter(
                            [pair[0] for pair in fit_trace],
                            [pair[1] for pair in fit_trace],
                            steps=32,
                            min_retention=0.2,
                        )
                        fit_trace.clear()
                        fitted = True
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
            # ⚠️ 必须重新取一次：`weight` 原来指向旧张量，不更新的话每一步都从**同一个**初始
            # 权重算起，更新永远累积不起来（门禁实测 |dW|/|W| = 0.0000 才发现的）。
            weight = harness.shared_weight

    # 可分性：用**全部训练标签**测表示本身（训练时规则只见到有标签子集，见规格）。
    train_features = features(train_inputs, harness.shared_weight)
    all_labels = labels.numpy()
    fisher = fisher_ratio(train_features, all_labels)
    probe = linear_probe_accuracy(train_features, all_labels)

    # 初始权重的对照：没有它，就分不清"规则改善了表示"与"随机投影本来就这样"。
    init_features = features(train_inputs, initial)
    fisher_init = fisher_ratio(init_features, all_labels)
    probe_init = linear_probe_accuracy(init_features, all_labels)

    test_features = features(test_inputs, harness.shared_weight)
    accuracy, n_labeled = _ncm_accuracy(
        train_features[mask], all_labels[mask], test_features, test.labels.numpy()
    )
    ridge_accuracy, _ = _ridge_accuracy(
        train_features[mask], all_labels[mask], test_features, test.labels.numpy()
    )
    delta_ratio = float((harness.shared_weight - initial).norm() / (initial.norm() + 1e-12))
    return {
        "arm": arm.name,
        "kind": arm.kind,
        "fraction": fraction,
        "seed": seed,
        "accuracy": accuracy,
        "ridge_accuracy": ridge_accuracy,
        "fisher": fisher,
        "probe": probe,
        "fisher_init": fisher_init,
        "probe_init": probe_init,
        "n_labeled": n_labeled,
        "alpha": float(harness.arbiter.alpha),
        "weight_delta_ratio": delta_ratio,
        "gate_passed": bool(delta_ratio >= GATE_RATIO),
        "shuffle_labels": shuffle_labels,
        "alpha_fitted": fitted,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="协同性基准：H / E / MIX")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--fractions", default="0.01")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--shared", type=int, default=256)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--arms", default="H,E,MIX", help="逗号分隔，可选 H / E / MIX")
    parser.add_argument("--fit-steps", type=int, default=200, help="MIX 臂用来学 α 的步数")
    parser.add_argument(
        "--no-fit-alpha",
        action="store_true",
        help="MIX 臂冻结在 alpha=0.5，不跑 ES 拟合（对照）",
    )
    parser.add_argument("--shuffle-labels", action="store_true", help="打乱训练标签（对照）")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="MIX 臂的初始/冻结 α（配 --no-fit-alpha 时就是固定配比；默认 0.5）",
    )
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    catalogue = {"H": Arm("H", "hebbian"), "E": Arm("E", "eprop"), "MIX": Arm("MIX", "mix")}
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
                    fit_steps=args.fit_steps,
                    fit_alpha=not args.no_fit_alpha,
                    alpha=args.alpha,
                    shuffle_labels=args.shuffle_labels,
                )
                results.append(row)
                gate = "PASS" if row["gate_passed"] else "FAIL"
                print(
                    f"  arm={arm.name:3s} seed={seed} "
                    f"Fisher={row['fisher']:.3f} (init {row['fisher_init']:.3f})  "
                    f"probe={row['probe']:.3f} (init {row['probe_init']:.3f})  "
                    f"|dW|/|W|={row['weight_delta_ratio']:.4f} gate={gate}"
                )

    summary = {
        "experiment": "arbiter synergy benchmark v2 (sMNIST, fixed permutation, few labels)",
        "chance": CHANCE,
        "gate_ratio": GATE_RATIO,
        "config": {"epochs": args.epochs, "batch": args.batch, "n_shared": args.shared},
        "rows": results,
        "repro": collect(
            "仲裁器协同性基准 v2",
            seeds=args.seeds,
            elapsed_s=time.perf_counter() - started,
            notes=[
                "固定随机置换；分层少量标签",
                "共用闭式岭回归读出；逐时间步整流特征；门禁 |dW|/|W| >= 0.05",
            ],
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
