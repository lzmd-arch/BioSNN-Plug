"""脉冲版 R-STDP Actor：LIF 决策 + 带 Δt 的 STDP 资格痕迹 + 成功信号调制。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

## 一个决策的完整过程

1. 群体编码后的状态（``(n_features,)``）在 **T 步窗口**内脉冲编码 → ``(T, n_features)``；
2. 喂给 LIF 输出层 → ``(T, n_actions)`` 的输出脉冲；
3. **本窗口内的 pre/post 配对**按 Δt 加权，得到 ``Δw``（:mod:`.plasticity`），累进资格痕迹；
4. 动作由窗口内**脉冲计数**决定（argmax 或 Boltzmann 采样）。

成功信号到达时：``w ← w + η·S·e``，然后按 L1 归一化每列（计划书 §3.2 的「权重归一化」，
防止突触动态失控），回合结束清空痕迹。

## 与速率版的差别，一句话

速率版把第 3 步写成 ``x_i · a_j``（同时刻乘积）——**Δt 被抹掉了**。这里第 3 步是真的
配对窗口，所以 ``τ_+`` / ``τ_−`` 是活的自由参数，而计划书 §3.2 里那句「TD-LTP 与 TD-STDP
的差别（仅计 pre-before-post）」**在这个实现里才有意义**。
"""

from __future__ import annotations

import torch

from research.rstdp_spiking.encoding import spike_encode
from research.rstdp_spiking.neurons import LIFLayer, LIFState
from research.rstdp_spiking.plasticity import EligibilityTrace, STDPWindow, pair_based_stdp

__all__ = ["SpikingActor"]


class SpikingActor:
    """离散动作的脉冲 Actor。

    Attributes:
        layer: 输出层（``n_features → n_actions`` 的 LIF）。
        trace: 跨决策累积的资格痕迹。
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        *,
        steps: int = 25,
        window: STDPWindow | None = None,
        tau: float = 20.0,
        threshold: float = 1.0,
        n_refractory: int = 2,
        learning_rate: float = 1e-3,
        trace_decay: float = 0.9,
        weight_norm: float = 1.0,
        normalize: bool = True,
        action_sampling: str = "boltzmann",
        logit_scale: float = 1.0,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        if steps <= 0:
            raise ValueError(f"steps 必须为正，收到 {steps}。")
        if weight_norm <= 0:
            raise ValueError(f"weight_norm 必须为正，收到 {weight_norm}。")
        if action_sampling not in ("epsilon_greedy", "boltzmann"):
            raise ValueError(
                f"action_sampling 只能是 'epsilon_greedy' 或 'boltzmann'，收到 {action_sampling!r}。"
            )
        if logit_scale <= 0:
            raise ValueError(f"logit_scale 必须为正，收到 {logit_scale}。")

        device = device or torch.device("cpu")
        self.n_features = int(n_features)
        self.n_actions = int(n_actions)
        self.steps = int(steps)
        self.window = window or STDPWindow()
        self.learning_rate = float(learning_rate)
        self.weight_norm = float(weight_norm)
        self.normalize = bool(normalize)
        self.action_sampling = action_sampling
        self.logit_scale = float(logit_scale)
        self.device = device

        self.layer = LIFLayer(
            n_features,
            n_actions,
            tau=tau,
            threshold=threshold,
            n_refractory=n_refractory,
            generator=generator,
        ).to(device)
        self.trace = EligibilityTrace(n_features, n_actions, decay=trace_decay, device=device)
        if self.normalize:
            self._normalize()

    # ------------------------------------------------------------------ 内部
    @torch.no_grad()
    def _normalize(self) -> None:
        """每列（一个动作神经元）的 L1 范数归到 ``weight_norm``。

        速率版同样这么做——计划书 §3.2 要求权重归一化来防「突触动态失控」。这里不归一化的话
        会更快失控：STDP 的 LTP 项随发放率平方增长。
        """
        column = self.layer.weight.abs().sum(dim=0, keepdim=True).clamp(min=1e-8)
        self.layer.weight.mul_(self.weight_norm / column)

    @torch.no_grad()
    def _simulate(self, pre_spikes: torch.Tensor) -> torch.Tensor:
        """跑完 T 步。进 ``(T, batch, n_features)``，出 ``(T, batch, n_actions)``。"""
        batch = pre_spikes.shape[1]
        state = LIFState.zeros(batch, self.n_actions, device=self.device)
        collected = []
        for t in range(self.steps):
            spikes, state = self.layer.step(pre_spikes[t], state)
            collected.append(spikes)
        return torch.stack(collected)

    def _window(self, features: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """一个决策窗口的输入脉冲，``(T, 1, n_features)``。"""
        return spike_encode(features, self.steps, generator=generator).unsqueeze(1)

    # ------------------------------------------------------------------ 对外
    @torch.no_grad()
    def counts(
        self, features: torch.Tensor, *, generator: torch.Generator, repeats: int = 1
    ) -> torch.Tensor:
        """跑窗口拿脉冲计数，**不碰痕迹**（评测走这条）。

        Args:
            repeats: 重复几个窗口取平均。脉冲网络的输出本身是随机的（输入编码就是伯努利
                采样），所以评测时重复几次能压低方差——这与速率版「贪心评测是确定性的」
                不同，是脉冲化的代价之一，写在 README 的边界里。
        """
        if repeats <= 0:
            raise ValueError(f"repeats 必须为正，收到 {repeats}。")
        total = torch.zeros(self.n_actions, device=self.device)
        for _ in range(repeats):
            total += self._simulate(self._window(features, generator)).sum(dim=0).squeeze(0)
        return total / repeats

    @torch.no_grad()
    def step(
        self,
        features: torch.Tensor,
        *,
        generator: torch.Generator,
        exploration: float = 0.0,
        greedy: bool = False,
        learn_trace: bool = True,
    ) -> int:
        """一个决策：跑窗口、按需累积痕迹、选动作。

        ``greedy=True`` 时无视采样方案直接返回脉冲计数最大的动作——**评测必须走这条路**。
        （速率版在这里踩过一个坑：boltzmann 分支不看 ``exploration``，于是「贪心评测」实际
        量的是一个随机策略。这里把贪心做成显式分支，不给那个坑留位置。）
        """
        pre_spikes = self._window(features, generator)
        post_spikes = self._simulate(pre_spikes)
        if learn_trace:
            self.trace.accumulate(pair_based_stdp(pre_spikes, post_spikes, self.window))

        spike_counts = post_spikes.sum(dim=0).squeeze(0)
        if greedy:
            return int(spike_counts.argmax().item())
        if self.action_sampling == "boltzmann":
            probabilities = torch.softmax(self.logit_scale * spike_counts, dim=-1)
            return int(torch.multinomial(probabilities, 1, generator=generator).item())
        if exploration > 0 and float(torch.rand(1, generator=generator).item()) < exploration:
            return int(torch.randint(self.n_actions, (1,), generator=generator).item())
        return int(spike_counts.argmax().item())

    @torch.no_grad()
    def apply_success_signal(self, success_signal: float) -> None:
        """``w ← w + η·S·e``，然后归一化。痕迹的清理由 :meth:`end_episode` 负责。"""
        self.layer.weight.add_(self.learning_rate * float(success_signal) * self.trace.value)
        if self.normalize:
            self._normalize()

    @torch.no_grad()
    def end_episode(self) -> None:
        self.trace.reset()

    @torch.no_grad()
    def trace_rms(self) -> float:
        return self.trace.rms()
