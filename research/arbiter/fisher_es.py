"""以 Fisher 可分性为目标的 ES（预注册见任务规格 §预注册）。

**为什么不是代理目标**：代理与验收脱节是这一类事故的根。整数冲突事件就是极端形态——
它在 α=0.5 附近恒为 0，ES 于是根本没在学。这里直接拿验收度量当目标，
每评估一个候选就从同一份权重快照出发跑一段**短程训练**，再在固定探针集上算 Fisher。

成本结构：单次评估 = `n_steps` 个训练步（两规则各一次）+ 一次 Fisher。
预注册的门槛是单次 ≤ 主导数单种子耗时的 5%，总预算 ≤ 2×。

⚠️ **实测结论（2026-10-24）：这条路在可承受的探针长度上没有区分度。**
8 / 50 / 200 / 800 步的探针上，Fisher 在 α ∈ {0.2 … 0.8} 上的极差是 0.026–0.039，
不随步数增长也不随 α 单调——因为完整训练要 37,480 步才到 0.65，几百步还停在随机投影那带。
本模块保留为**已否决方案的实现**，不要再拿去跑基准。
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["fisher_es_fit"]

from research.arbiter.separability import fisher_ratio


def _blend_apply(
    harness,
    *,
    alpha: float,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    labeled: np.ndarray,
    hebbian_update,
    eprop_update,
) -> None:
    """按 α 混合两规则的一步更新并写回共享权重。（与 runner 主循环同一套语义）"""
    delta_h = hebbian_update(harness.hebbian, batch_x, batch_y)
    if labeled.any():
        delta_e = eprop_update(
            harness.eprop, batch_x[:, torch.as_tensor(labeled)], batch_y[labeled]
        )
    else:
        delta_e = torch.zeros_like(harness.shared_weight)
    blended = harness.arbiter.blend(
        delta_e.detach().cpu().numpy(), delta_h.detach().cpu().numpy().T
    )
    new_weight = harness.shared_weight.detach().cpu() + torch.as_tensor(
        blended, dtype=torch.float32
    )
    harness._shared = new_weight.to(harness.shared_weight.device)
    harness.sync_from_shared()


def fisher_es_fit(
    harness,
    *,
    train_inputs: torch.Tensor,
    labels: torch.Tensor,
    mask: np.ndarray,
    probe_inputs: torch.Tensor,
    probe_labels: np.ndarray,
    features,
    hebbian_update,
    eprop_update,
    evaluations: int = 32,
    probe_steps: int = 8,
    probe_batch: int = 64,
    min_retention: float = 0.2,
    sigma: float = 0.1,
    seed: int = 0,
) -> tuple[float, list[tuple[float, float]]]:
    """在 `alpha` 上做 (1+1)-ES，目标是**探针 Fisher 比**。

    Args:
        harness: 已建好的共享突触 harness（当前权重即快照）。
        train_inputs / labels / mask: 短程训练用的数据与「哪些样本有标签」。
        probe_inputs / probe_labels: 算 Fisher 的固定探针集（**不参与短程训练**）。
        features: 把 ``(T, n, n_in)`` 序列 + 权重 → ``(n, d)`` 特征的可调用对象。
        hebbian_update / eprop_update: runner 里的两个更新函数（保持语义一致）。
        evaluations: 评估预算（预注册固定 32，不按时间停）。
        probe_steps: 每个候选取多少步短训练。
        min_retention: α 下界（硬约束）。
        sigma: ES 扰动尺度。
        seed: 扰动序列的种子——**复现的前提**。

    Returns:
        ``(best_alpha, trace)``，trace 是 ``(alpha, fisher)`` 的评估历史。
    """
    snapshot = harness.shared_weight.detach().clone()
    rng = np.random.default_rng(seed)
    n = train_inputs.shape[1]
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))

    def evaluate(alpha: float) -> float:
        harness._shared = snapshot.clone()
        harness.sync_from_shared()
        for step in range(probe_steps):
            idx = order[(step * probe_batch) % n : (step * probe_batch) % n + probe_batch]
            _blend_apply(
                harness,
                alpha=alpha,
                batch_x=train_inputs[:, idx],
                batch_y=labels[idx],
                labeled=mask[idx.numpy()],
                hebbian_update=hebbian_update,
                eprop_update=eprop_update,
            )
        probe_features = features(probe_inputs, harness.shared_weight)
        score = float(fisher_ratio(probe_features, probe_labels))
        harness._shared = snapshot.clone()
        harness.sync_from_shared()
        return score

    trace: list[tuple[float, float]] = []
    alpha = float(harness.arbiter.alpha)
    best = evaluate(alpha)
    trace.append((alpha, best))

    for _ in range(max(0, evaluations - 1)):
        trial = float(np.clip(alpha + rng.normal(0.0, sigma), min_retention, 1.0))
        if trial == alpha:
            continue
        score = evaluate(trial)
        trace.append((trial, score))
        if score > best:
            best, alpha = score, trial

    harness.arbiter.alpha = alpha
    harness._shared = snapshot.clone()
    harness.sync_from_shared()
    return alpha, trace
