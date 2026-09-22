"""计划书 §9 的网络健康指标。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

指标口径只写一遍，三个验证线共用——**两条线各报一个"活跃神经元比例"，读者会以为
它们可比**。所以这里把定义钉死，并说明为什么是这么定的。

## 活跃神经元比例

计划书 §9 要求在第二阶段 > 60%、第四阶段 > 80%；§3.1 的「死亡神经元防护」把
**低于 60% 触发阈值调整**作为硬性动作。§七 第一阶段成功标准里的「活跃神经元比例
> 60%」就是这条。

**定义**：在观察窗内**至少发放过一次**的神经元占全部神经元的比例。

选"至少一次"而不是"平均发放率高于某个值"，是因为这个指标要回答的问题是
**这个神经元还活着吗**——一个偶尔发放的神经元仍然参与编码，一个从不发放的神经元
则占着参数容量却毫无贡献。用发放率阈值会把前者误判为死亡，而阈值取多少纯属约定，
不如让判据落在"有没有"这个二值事实上。

**观察窗**取整个评估集，不取单个 batch：单个 batch 上神经元不发放很正常，据此判定
死亡会导致阈值被无谓地调低。

## 脉冲稀疏度

计划书 §9 要求在第二阶段 > 90%、第四阶段 > 98%。

**定义**：脉冲张量里零元素的占比，即 ``1 - 非零占比``。与
:attr:`biosnn_bus.SpikeTrain.density` 互为补数——那条实现定义的是稠密度，这里
报的是稀疏度，两者刻意都留着，换算关系写在这里以免有人当成两回事。
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "DEAD_NEURON_THRESHOLD",
    "active_neuron_count",
    "active_neuron_fraction",
    "spike_density",
    "spike_sparsity",
    "summarize",
]

#: §3.1：活跃神经元比例低于此值时触发放电阈值调整。
DEAD_NEURON_THRESHOLD = 0.60


def _as_2d(spikes: Any) -> np.ndarray:
    """接受 ``(T, N)`` 的 numpy / torch / SpikeTrain，统一成 numpy。

    非布尔输入按"大于 0 即为发放"处理（发放率或膜电位代理都适用）。
    """
    # SpikeTrain 有 .data；torch.Tensor 有 .detach
    data = getattr(spikes, "data", spikes)
    detach = getattr(data, "detach", None)
    if detach is not None:
        data = detach().to("cpu").numpy()
    array = np.asarray(data)

    if array.ndim != 2:
        raise ValueError(
            f"指标需要 (T, N) 的二维脉冲张量，收到形状 {array.shape}。"
            f"若只有一个时间步，请显式写成 (1, N)。"
        )
    return array


def _fired(array: np.ndarray) -> np.ndarray:
    """``(T, N)`` 的布尔矩阵，True 表示该时刻该神经元发放。"""
    if np.issubdtype(array.dtype, np.bool_):
        return array
    # 浮点输入可能带极小负值（膜电位代理），用 > 0 而不是 != 0
    return array > 0


def active_neuron_count(spikes: Any) -> int:
    """观察窗内**至少发放过一次**的神经元个数。"""
    return int(_fired(_as_2d(spikes)).any(axis=0).sum())


def active_neuron_fraction(spikes: Any) -> float:
    """活跃神经元比例，落在 ``[0, 1]``。

    低于 :data:`DEAD_NEURON_THRESHOLD` 应触发 §3.1 的阈值调整。
    """
    fired = _fired(_as_2d(spikes))
    return float(fired.any(axis=0).mean())


def spike_density(spikes: Any) -> float:
    """非零元素占比，落在 ``[0, 1]``。"""
    fired = _fired(_as_2d(spikes))
    return float(fired.mean())


def spike_sparsity(spikes: Any) -> float:
    """脉冲稀疏度 = ``1 - 密度``，落在 ``[0, 1]``。"""
    return 1.0 - spike_density(spikes)


def summarize(spikes: Any) -> dict[str, float | int]:
    """一次算齐评估与记录要用的量。

    返回的键名与复现记录、主题 README 里的用法保持一致。
    """
    array = _as_2d(spikes)
    return {
        "n_steps": int(array.shape[0]),
        "n_neurons": int(array.shape[1]),
        "active_neurons": active_neuron_count(array),
        "active_neuron_fraction": active_neuron_fraction(array),
        "spike_sparsity": spike_sparsity(array),
        "spike_density": spike_density(array),
    }
