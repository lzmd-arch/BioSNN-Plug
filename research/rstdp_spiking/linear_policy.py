"""D2 的线性策略适配器：把速率版 ``RSTDPActor`` 接进脉冲 harness。

**这不是候选最终形态，是 sanity control。** 预注册见
``sweep_results/rstdp_spiking_policy_control_prereg.md``：同一套环境、群体 Critic、
种子与训练循环，只把"动作怎么选"换成速率版已验证的线性 Boltzmann 配方。

复用 ``research.rstdp.RSTDPActor``（**不复制**它的 ``trace_center`` 数学）：本类只负责
把它的接口对齐到 :class:`~research.rstdp_spiking.actor.SpikingActor` 的三个方法——
``step`` / ``apply_success_signal`` / ``end_episode``——好让 ``cartpole.train`` 的循环
一行不改地跑两种策略。
"""

from __future__ import annotations

import torch

from research.rstdp import RSTDPActor

__all__ = ["LinearRatePolicy"]


class LinearRatePolicy:
    """``logits = W·x``、Boltzmann 采样、``x⊗(a−π)`` 痕迹；接口对齐 SpikingActor。"""

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        *,
        learning_rate: float = 1e-3,
        trace_decay: float = 0.9,
        logit_scale: float = 1.0,
        normalize: bool = True,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.inner = RSTDPActor(
            n_features,
            n_actions,
            learning_rate=learning_rate,
            trace_decay=trace_decay,
            normalize=normalize,
            action_sampling="boltzmann",
            logit_scale=logit_scale,
            trace_center="sampling",
            device=device,
            generator=generator,
        )
        self.last_entropy = 0.0

    # ---------------------------------------------------------------- 与 actor 对齐
    @property
    def logit_scale(self) -> float:
        return self.inner.logit_scale

    @logit_scale.setter
    def logit_scale(self, value: float) -> None:
        self.inner.logit_scale = float(value)

    def sampling_scale(self) -> float:
        return self.inner.logit_scale

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
        del exploration  # Boltzmann 下探索由 logit_scale 决定，与 SpikingActor 一致
        action = self.inner.select_action(features, generator=generator, greedy=greedy)
        if learn_trace:
            self.inner.accumulate_trace(features, action)
        return action

    @torch.no_grad()
    def apply_success_signal(self, success_signal: float) -> None:
        self.inner.apply_success_signal(float(success_signal))

    @torch.no_grad()
    def end_episode(self) -> None:
        self.inner.reset_trace()

    @property
    def episode_entropy(self) -> float:
        # 线性策略的熵没有接进训练日志；D2 不记熵，返回 0 而不是伪造一个值。
        return 0.0

    def trace_rms(self) -> float:
        return self.inner.trace_rms()

    # ---------------------------------------------------------------- 权重快照
    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"weights": self.inner.weights.detach().clone()}

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.inner.weights.copy_(state["weights"])
        # polyak 未启用时 weights_deployed 与 weights 是同一个张量；这里显式同步，
        # 免得未来加了 polyak 时静默地"评测旧权重"。
        self.inner.weights_deployed.copy_(state["weights"])
