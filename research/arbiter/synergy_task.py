"""协同性基准的任务数据：固定置换 + 少量标签。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

任务规格：[`sweep_results/arbiter_synergy_task_spec.md`](../../sweep_results/arbiter_synergy_task_spec.md)。

## 两件事，都只做一件事

1. **固定随机置换**：把每一步的 28 个输入维打乱（顺序固定、种子入配置）。
   它破坏"像素相邻 ⇒ 语义相邻"的局部结构，使纯 Hebbian 的相关结构不再对准标签。
2. **分层少量标签**：按类别分层抽出 `fraction` 的样本作为有标签集，
   其余样本**仍然参与训练**，但对 e-prop 不可见（它是有教师规则）。

## ⚠️ 共享点的维度

sMNIST 每步只喂一行像素，所以"输入 → 隐藏"这一处共享映射是 **28 → 256**，
不是静态 MNIST 的 784 → 256。两条线都能自然使用这个形状：
e-prop 的 `n_in=28`、Hebbian 首层的 `28 → 256`。**这不是缩小模型，是任务形态决定的。**
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["label_mask", "make_permutation", "permute_inputs"]


def make_permutation(n_in: int, *, seed: int) -> np.ndarray:
    """返回一个长度为 ``n_in`` 的固定置换（0..n_in-1 的打乱）。"""
    if n_in <= 0:
        raise ValueError(f"n_in 必须为正，收到 {n_in}。")
    return np.random.default_rng(seed).permutation(n_in)


def permute_inputs(inputs: torch.Tensor, permutation: np.ndarray) -> torch.Tensor:
    """沿**最后一维**（每步的输入特征）施加置换。

    Args:
        inputs: ``(T, n, n_in)`` 的序列张量。
        permutation: :func:`make_permutation` 的输出。
    """
    if inputs.ndim != 3:
        raise ValueError(f"inputs 必须是 (T, n, n_in)，收到 {tuple(inputs.shape)}。")
    perm = np.asarray(permutation)
    if perm.shape != (inputs.shape[-1],):
        raise ValueError(f"置换长度 {perm.shape} 与输入维度 {inputs.shape[-1]} 不一致。")
    index = torch.as_tensor(perm, dtype=torch.long, device=inputs.device)
    return inputs.index_select(-1, index)


def label_mask(labels, fraction: float, *, seed: int) -> np.ndarray:
    """按类别**分层**抽出 ``fraction`` 的样本，返回布尔掩码。

    Args:
        labels: ``(n,)`` 的类别下标（numpy 或 tensor）。
        fraction: 有标签比例，``(0, 1]``。``1.0`` 时全为 ``True``。
        seed: 抽样种子。

    Returns:
        ``(n,)`` 的布尔数组。

    Note:
        每类至少抽 1 个（只要该类非空），否则 1% 档会有类别一个标签都没有——
        那会让"E 单独失败"变成算术上的必然，而不是任务设计的结果。
    """
    y = np.asarray(labels if not isinstance(labels, torch.Tensor) else labels.cpu())
    if y.ndim != 1:
        raise ValueError(f"labels 必须是一维，收到 {y.shape}。")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction 应落在 (0, 1]，收到 {fraction}。")

    rng = np.random.default_rng(seed)
    mask = np.zeros(y.shape[0], dtype=bool)
    if fraction >= 1.0:
        mask[:] = True
        return mask

    for label in np.unique(y):
        index = np.flatnonzero(y == label)
        take = max(1, round(fraction * index.size))
        mask[rng.choice(index, size=min(take, index.size), replace=False)] = True
    return mask
