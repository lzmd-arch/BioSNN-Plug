"""线性 TD 群体 Critic（脉冲执行层的成功信号来源）。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

## 为什么必须有它

计划书 §3.2 引 Frémaux et al. (2010)：R-STDP 的成功偏移达 σR 的约 25% 就无法学习，
偏移 < −0.4σR 时性能甚至低于学习前；调 STDP 窗口或换权重依赖模型都解决不了，
结构性解法是**刺激特异性奖励预测**。

用回报的滑动平均当基线**不够**——它对所有状态给出同一个基线，而正确要预测的是
"**这个状态**值多少"。第一版就是这么做的，400 回合后停在随机水平（见
`research/rstdp_spiking/README.md` 的执行记录）。

## 更新规则（纯局部，无梯度回传）

```text
V(s) = w · x(s)
δ    = r + γ·V(s′) − V(s)        （终止状态不加 V(s′)）
w   ← w + β·δ·x(s)
```

这是 TD(0) 加**线性**函数逼近：Critic 的每个权重只用到自己的输入活动与一个标量误差 δ，
不走链式法则，与 Actor 的局部性同一标准。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = ["LinearCritic"]


@dataclass
class LinearCritic:
    """线性 TD(0) Critic。

    Attributes:
        weights: ``(n_features,)`` 的值函数权重。
        gamma: 折扣因子。
        learning_rate: Critic 自己的步长 ``β``（与 Actor 的 η 分开）。
    """

    n_features: int
    gamma: float = 0.99
    learning_rate: float = 1e-2
    weights: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        if self.n_features <= 0:
            raise ValueError(f"n_features 必须为正，收到 {self.n_features}。")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma 应在 [0, 1] 内，收到 {self.gamma}。")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate 必须为正，收到 {self.learning_rate}。")
        self.weights = torch.zeros(self.n_features)

    def value(self, features: torch.Tensor) -> float:
        """``V(s)``。"""
        return float(torch.dot(self.weights, features.detach().flatten()))

    def update(
        self,
        features: torch.Tensor,
        reward: float,
        next_features: torch.Tensor,
        *,
        terminated: bool,
    ) -> float:
        """做一次 TD(0) 更新，返回 δ（Actor 的成功信号）。"""
        x = features.detach().flatten()
        x_next = next_features.detach().flatten()
        delta = float(reward) + (0.0 if terminated else self.gamma * self.value(x_next))
        delta -= self.value(x)
        self.weights.add_(self.learning_rate * delta * x)
        return delta

    def reset(self) -> None:
        """清空权重（换回合序列时用）。"""
        self.weights.zero_()
