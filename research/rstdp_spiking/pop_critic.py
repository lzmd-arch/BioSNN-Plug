"""TD-LTP 群体 Critic（Frémaux et al. 2013 的结构，脉冲目录内的实现）。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

## 结构

论文里的 Critic 是一**群**神经元，值由它们的活动经**固定**权重读出。本实现：

```text
y_j   = σ( g·(w_j · x − b) )        第 j 个单元的活动 ∈ (0, 1)，b 固定
V(s)  = Σ_j u_j·y_j + out_bias      u 固定（均匀）
e_ji ← λ·e_ji + x_i·y_j             第二因子是**单元自己的活动**，有界且非负
w_j  ← w_j + η·δ·e_j，再归一化 ‖w_j‖₂ = 1
```

## 为什么不是单权重线性 Critic

`LinearCritic` 只有一条权重向量，等于把"单元输出"和"值"当成同一个量：

1. **正值反馈**：`e = x·V` 且 `V = w·x` → 更新变大、`V` 更大、痕迹再变大；
2. **尺度自由度被归一化抹掉**：只能旋转、不能设定输出尺度，`δ` 容易退化成常数。

群体结构两条都治：第二因子有界非负，正值反馈断掉；值的尺度由固定的 `u` 承担，
`w_j` 只管形状，行归一化不再夺走自由度。

## 与速率版的关系

`research/rstdp/td_ltp.py` 里有一个功能更全的 `PopulationCritic`（含 bias 学习、有符号读出、
gradient 后因子等）。**本目录不 import 它**（仓库约定：各主题只共享 `research/common/`），
这里是一个**精简移植**：只保留 CartPole 这条线用得到的部分，参数默认值按本线的量纲取。

⚠️ **这不是等价实现**：速率版里针对"终止步稳定拿到大负 δ"的 `output_bias` 学习、以及
有符号读出的限制，这里都**没有**——诊断那两个病是速率版的历史，脉冲线还没走到那一步。
"""

from __future__ import annotations

import torch

__all__ = ["PopulationCritic", "td_error"]


def td_error(
    reward: float, value: float, next_value: float, *, terminated: bool, gamma: float
) -> float:
    """``δ = r + γ·V(s′) − V(s)``；终止状态不加 ``V(s′)``。"""
    target = float(reward) + (0.0 if terminated else gamma * next_value)
    return target - value


class PopulationCritic:
    """群体 + 固定读出的 Critic。

    Attributes:
        weights: ``(n_units, n_features)``，每行是一个单元的 ``w_j``。
        trace: 同形状的资格痕迹。
        readout: ``(n_units,)``，固定均匀读出。
    """

    def __init__(
        self,
        n_features: int,
        n_units: int = 64,
        *,
        learning_rate: float = 5e-3,
        trace_decay: float = 0.9,
        gain: float = 8.0,
        bias: float = 0.4,
        value_scale: float = 200.0,
        gamma: float = 0.99,
        bias_learning_rate: float = 1e-2,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        if n_features <= 0 or n_units < 1:
            raise ValueError(f"n_features 与 n_units 必须为正，收到 {n_features}、{n_units}。")
        if not 0.0 <= trace_decay < 1.0:
            raise ValueError(f"trace_decay 应在 [0, 1) 内，收到 {trace_decay}。")
        if gain <= 0 or value_scale <= 0:
            raise ValueError(f"gain 与 value_scale 必须为正，收到 {gain}、{value_scale}。")

        device = device or torch.device("cpu")
        self.n_features = int(n_features)
        self.n_units = int(n_units)
        self.learning_rate = float(learning_rate)
        self.trace_decay = float(trace_decay)
        self.gain = float(gain)
        self.bias = float(bias)
        self.value_scale = float(value_scale)
        self.gamma = float(gamma)
        self.bias_learning_rate = float(bias_learning_rate)

        gen = generator or torch.Generator(device=device)
        weights = torch.randn(n_units, n_features, generator=gen, device=device)
        self.weights = weights / (weights.norm(dim=1, keepdim=True) + 1e-12)
        self.trace = torch.zeros_like(self.weights)
        self.readout = torch.full((n_units,), value_scale / n_units, device=device)
        # 输出侧可学偏置。**不是**单元阈值：它平移的是值本身。
        # 没有它，群体读出到不了"临死那步真值≈0"，终止步会稳定拿到一个大负 δ，
        # 而那一次更新落在痕迹视野内、与动作无关（速率版记载为 −11.3 vs 内部 +0.07）。
        self.out_bias = torch.zeros((), device=device)

    # ------------------------------------------------------------------ 前向
    def rates(self, features: torch.Tensor) -> torch.Tensor:
        """``(n_units,)`` 的单元活动。"""
        x = features.detach().flatten()
        return torch.sigmoid(self.gain * (self.weights @ x - self.bias))

    def value(self, features: torch.Tensor) -> float:
        return float(torch.dot(self.readout, self.rates(features)) + self.out_bias)

    # ------------------------------------------------------------------ 学习
    def update(
        self,
        features: torch.Tensor,
        reward: float,
        next_features: torch.Tensor,
        *,
        terminated: bool,
    ) -> float:
        """TD-LTP 一步：累积痕迹 → 用 δ 调制权重。返回 δ（同时是 Actor 的成功信号）。"""
        x = features.detach().flatten()
        y = self.rates(features)
        delta = td_error(
            reward,
            self.value(features),
            self.value(next_features),
            terminated=terminated,
            gamma=self.gamma,
        )
        self.trace.mul_(self.trace_decay).add_(torch.outer(y, x))
        self.weights.add_(self.learning_rate * delta * self.trace)
        norms = self.weights.norm(dim=1, keepdim=True) + 1e-12
        self.weights.div_(norms)
        self.out_bias.add_(self.bias_learning_rate * delta)
        return delta

    def reset_trace(self) -> None:
        """清空痕迹（回合边界用；**权重保留**）。"""
        self.trace.zero_()
