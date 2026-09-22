"""TD-LTP：Critic 的学习规则（Frémaux et al. 2013, Eq. 17）。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

出处由 **ADR-0007** 结项（计划书 §十一 的 P0 待办）。原文逐字：

> Because it has, roughly, the form of "TD error signal × Hebbian LTP", we call this
> learning rule **TD-LTP**.

规则本体（三因子）：

    Δw_i ∝ δ(t) · κ ∗ [ pre_i(t) · post(t) ]      ← 符合窗**仅计 pre-before-post**

## 三因子各是什么

| 因子 | 内容 | 局部性 |
| :--- | :--- | :--- |
| 1 | 前突触脉冲串 | 突触本地 |
| 2 | 后突触脉冲串 | 突触本地 |
| 3 | **δ(t)，全局标量** TD 误差 | 经多巴胺式广播到达每个突触 |

原文对第三因子的说明（这句是"纯局部"主张的依据）：

> The first one is the TD error term, which is the same for all synapses, and can thus be
> considered as a global factor, possibly transmitted by one or more neuromodulators.

**非局部量只有这一个标量**，不含误差向量的反向传播。

## 与 TD-STDP 的实质差别

原文 Figure 2A 的图注：

> Bottom: TD-STDP is a TD-modulated variant of R-STDP. The main difference with TD-LTP is
> the presence of a post-before-pre component in the coincidence window.

即：**TD-LTP 的符合窗只计 pre-before-post，TD-STDP 还含 post-before-pre 分量。**
ADR-0007 的备选 C 就是退回 TD-STDP（改动仅此一处），若本实现在 CartPole 上受阻则启用，
并另开 ADR。

## 速率型实现下的形式

本项目用速率型单元（CartPole 的状态是连续量，群体编码后做速率读出），所以"pre 脉冲串与
post 脉冲串的符合"落地为两者的乘积，再经指数核 κ 滤波成资格痕迹：

    e_i(t) = λ·e_i(t−1) + x_i(t)·y(t)          λ = exp(−dt/τ_κ)
    Δw_i   = η · δ(t) · e_i(t)

**这是一个化简**：脉冲串之间的"符合窗"在速率型下退化成同时刻的乘积，"仅计
pre-before-post"这一条因此不再是可区分的结构（速率型下没有"先后"）。所以本模块
**无法**体现 TD-LTP 与 TD-STDP 的那处差别——如果将来要退回 TD-STDP，那处差别需要
在脉冲型的实现里才谈得上。这一点写进结论的边界，不含糊。
"""

from __future__ import annotations

import torch

__all__ = ["TDLCritic"]


class TDLCritic:
    """线性 Critic，用 TD-LTP 训练。

    ``V(s) = Σ_i v_i · x_i``（``x`` 是状态编码的活动），按

        e_i ← λ·e_i + x_i·V
        v_i ← v_i + η·δ·e_i

    更新。``δ`` 由外部给出（见 :func:`td_error`），本类只管"用 δ 调制资格痕迹"这件事。

    Attributes:
        weights: ``(n_features,)`` 的 ``v``。
        trace: ``(n_features,)`` 的资格痕迹 ``e``。
    """

    def __init__(
        self,
        n_features: int,
        *,
        learning_rate: float = 1e-3,
        trace_decay: float = 0.9,
        weight_decay: float = 0.0,
        init_scale: float = 0.01,
        weight_norm: float | None = 1.0,
        value_scale: float = 60.0,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        if not 0.0 <= trace_decay < 1.0:
            raise ValueError(f"trace_decay 应在 [0, 1) 内，收到 {trace_decay}。")
        if init_scale <= 0:
            raise ValueError(f"init_scale 必须为正，收到 {init_scale}。")
        if weight_norm is not None and weight_norm <= 0:
            raise ValueError(f"weight_norm 必须为正或 None，收到 {weight_norm}。")
        if value_scale <= 0:
            raise ValueError(f"value_scale 必须为正，收到 {value_scale}。")

        self.n_features = int(n_features)
        self.learning_rate = float(learning_rate)
        self.trace_decay = float(trace_decay)
        self.weight_decay = float(weight_decay)
        self.weight_norm = None if weight_norm is None else float(weight_norm)
        self.value_scale = float(value_scale)
        device = device or torch.device("cpu")

        # **不能用零初始化。** 资格痕迹是 ``e ← λ·e + x·V``，即被 Critic 自己的输出
        # ``V`` 门控；若 ``V ≡ 0``（零初始化且不更新），痕迹恒为 0，权重更新量
        # ``η·δ·e`` 也就恒为 0——Critic 永远学不动。这是三因子结构里"后突触活动作为
        # 第二因子"带来的**零点锁死**，不是调参能绕过的。所以这里用小幅随机初始化，
        # 与论文的随机初始化一致。
        #
        # **还要做权重归一化**（计划书 §3.2 那一句「每个神经元层面保持权重总和恒定，
        # 防止突触动态失控」）。不做的话上面那个正值反馈会失控：V 变大 → e = x·V 变大
        # → Δw 变大 → V 更大。实测就是权重发散成 NaN。把 ||w||₂ 钉住，V 就被
        # ``value_scale · ‖x‖`` 界住，反馈回路断开。
        #
        # ``value_scale`` 用来恢复被归一化拿掉的那个自由度（Critic 的整体增益）：
        # CartPole 的回报可达数百，而 ‖x‖ ≈ √N，所以需要几十倍的放大。
        generator = generator or torch.Generator(device=device)
        self.weights = torch.randn(n_features, device=device, generator=generator) * init_scale
        self.trace = torch.zeros(n_features, device=device)
        self._normalize()

    @torch.no_grad()
    def _normalize(self) -> None:
        """把权重向量归一到 L2 范数 ``weight_norm``（计划书 §3.2 的权重归一化）。"""
        if self.weight_norm is None:
            return
        norm = self.weights.norm().clamp_min(1e-12)
        self.weights.mul_(self.weight_norm / norm)

    @torch.no_grad()
    def value(self, features: torch.Tensor) -> torch.Tensor:
        """``V(s) = value_scale · (w·x)``。

        ``features`` 是 ``(n_features,)`` 或 ``(batch, n_features)``。
        """
        return self.value_scale * (features @ self.weights)

    @torch.no_grad()
    def update(self, features: torch.Tensor, value: torch.Tensor, td_error: float) -> None:
        """累积资格痕迹并用 ``δ`` 调制，就地更新权重。

        Args:
            features: ``(n_features,)`` 当前状态的活动。
            value: 标量 ``V(s)``（本步的预测）。
            td_error: 全局标量 ``δ``。
        """
        # 第二因子用**缩放前**的活动 ``w·x``，不是 ``V`` 本身：否则 trace 会随
        # ``value_scale`` 一起放大，把增益重复计一次。
        activity = value / self.value_scale
        self.trace.mul_(self.trace_decay).add_(features * activity)
        self.weights.add_(self.learning_rate * td_error * self.trace)
        if self.weight_decay:
            self.weights.mul_(1.0 - self.weight_decay)
        self._normalize()

    @torch.no_grad()
    def reset_trace(self) -> None:
        """回合结束时清空痕迹——资格痕迹不该跨回合累积。"""
        self.trace.zero_()


def td_error(
    reward: float,
    value: torch.Tensor,
    next_value: torch.Tensor,
    *,
    discount: float = 1.0,
) -> float:
    """一步 TD 误差 ``δ = r + γ·V(s') − V(s)``。

    ``value`` 与 ``next_value`` 是标量张量。回合终止时 ``V(s')`` 应传 0。
    """
    return float(reward + discount * float(next_value) - float(value))
