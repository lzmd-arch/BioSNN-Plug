"""R-STDP：Actor 的学习规则与权重归一化（计划书 §3.2）。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

    Δw_ij = STDP(Δt_ij) · S         （计划书 §3.2 的写法）

其中 ``S`` 是**无偏的成功信号**。Frémaux et al. (2013) 对这个结构的描述：

> In R-STDP, the effects of classic STDP are stored into an exponentially decaying, medium
> term, synapse-specific memory, called an eligibility trace. This trace is only imprinted
> into the actual synaptic weights when a global, neuromodulatory success signal is sent to
> the synapses. In R-STDP, the neuromodulatory signal is the reward minus a baseline,
> i.e., [R − b]. It was shown that for R-STDP to maximize reward, **the baseline must
> precisely match the mean (or expected) reward**.

最后那句就是"Critic 不是可选项"的来源：计划书 §3.2 引 Frémaux et al. (2010) 指出，成功
偏移达 σR 的约 25% 就无法学习，偏移 < −0.4σR 时性能甚至低于学习前。这个敏感性是通用
特性，调 STDP 窗口或换权重依赖模型都解决不了——**唯一的结构性解法是刺激特异性奖励预测**。

## 成功信号取哪一个

两种都实现，由调用方选：

* ``S = R − V(s)``（奖励预测误差）——计划书 §3.2 的读法（"S = R − ⟨R⟩"，Critic 给出 ⟨R⟩）；
* ``S = δ = R + γV(s') − V(s)``（TD 误差）——Frémaux 2013 指出的自然延伸：既然 δ 也是
  奖励预测误差信号，用它替代 S 就把 R-STDP 变成 TD 调制的规则。

两者都无偏（只要 ``V`` 收敛到期望奖励）。差别在视野：前者只看当下，后者往前看一步。

## 速率型下的落地

离散动作的 Actor：每个动作一个单元，后突触活动取**被选中动作的指示量**
``a_j ∈ {0, 1}``。资格痕迹按

    e_ij ← λ·e_ij + x_i · a_j

累积，权重按 ``w_ij ← w_ij + η·S·e_ij`` 更新。这与"经典 STDP 的效果先存进痕迹、只在成功
信号到达时写入权重"是同构的。

**这同样是化简**：速率型下没有"脉冲先后"可言，所以 `STDP(Δt)` 的具体窗口形状在这里退化
成"同时刻的乘积"。窗口参数（τ_+、τ_−）因此不是本实现的自由参数——这是速率型的代价。
"""

from __future__ import annotations

import torch

__all__ = ["RSTDPActor"]


class RSTDPActor:
    """离散动作的 R-STDP Actor，带逐神经元权重归一化。

    Attributes:
        weights: ``(n_features, n_actions)``。
        trace: 同形状的资格痕迹。
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        *,
        learning_rate: float = 1e-2,
        trace_decay: float = 0.9,
        weight_norm: float = 1.0,
        normalize: bool = True,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        if not 0.0 <= trace_decay < 1.0:
            raise ValueError(f"trace_decay 应在 [0, 1) 内，收到 {trace_decay}。")
        if weight_norm <= 0:
            raise ValueError(f"weight_norm 必须为正，收到 {weight_norm}。")

        device = device or torch.device("cpu")
        self.n_features = int(n_features)
        self.n_actions = int(n_actions)
        self.learning_rate = float(learning_rate)
        self.trace_decay = float(trace_decay)
        self.weight_norm = float(weight_norm)
        self.normalize = bool(normalize)

        generator = generator or torch.Generator(device=device)
        self.weights = torch.randn(n_features, n_actions, device=device, generator=generator)
        self.trace = torch.zeros(n_features, n_actions, device=device)
        if self.normalize:
            self._normalize()

    @torch.no_grad()
    def _normalize(self) -> None:
        """把每个动作单元（每一列）的入权重 **L1 范数**归一到 ``weight_norm``。

        计划书 §3.2：「每个神经元层面保持权重总和恒定，防止突触动态失控」。取 L1 而不是
        L2：L1 约束的是"权重总和"，与原文措辞一致；L2 约束的是能量，在稀疏编码下会让
        个别权重吃掉全部预算。
        """
        column_sums = self.weights.abs().sum(dim=0, keepdim=True).clamp_min(1e-12)
        self.weights.mul_(self.weight_norm / column_sums)

    @torch.no_grad()
    def scores(self, features: torch.Tensor) -> torch.Tensor:
        """各动作的得分，``(n_actions,)``。"""
        return features @ self.weights

    @torch.no_grad()
    def select_action(
        self,
        features: torch.Tensor,
        *,
        generator: torch.Generator,
        exploration: float = 0.0,
        return_explored: bool = False,
    ):
        """按得分选动作，``exploration`` 概率下随机探索。

        ``return_explored=True`` 时返回 ``(action, explored)``——调用方需要知道这一步
        是不是探索步（探索步不该拿去强化 Actor），而**只有这里知道**：随机数在这里被消耗，
        外面无法复现。
        """
        explored = False
        if exploration > 0 and float(torch.rand(1, generator=generator).item()) < exploration:
            action = int(torch.randint(self.n_actions, (1,), generator=generator).item())
            explored = True
        else:
            action = int(self.scores(features).argmax().item())
        return (action, explored) if return_explored else action

    @torch.no_grad()
    def update(self, features: torch.Tensor, action: int, success_signal: float) -> None:
        """累积资格痕迹，并用成功信号调制后写入权重。

        Args:
            features: ``(n_features,)`` 当前状态的活动。
            action: 被选中的动作下标。
            success_signal: 全局标量 ``S``（见模块 docstring 对两个取法的说明）。
        """
        indicator = torch.zeros(self.n_actions, device=self.weights.device)
        indicator[action] = 1.0
        self.trace.mul_(self.trace_decay).add_(torch.outer(features, indicator))
        self.weights.add_(self.learning_rate * success_signal * self.trace)
        if self.normalize:
            self._normalize()

    @torch.no_grad()
    def reset_trace(self) -> None:
        """回合结束时清空痕迹。"""
        self.trace.zero_()

    @torch.no_grad()
    def trace_rms(self) -> float:
        """痕迹的均方根，用于观察它有没有失控（计划书 §3.2 的"突触动态失控"）。"""
        return float(self.trace.pow(2).mean().sqrt().item())
