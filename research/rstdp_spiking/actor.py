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

from research.cognitive import ArousalChannel
from research.rstdp_spiking.neurons import LIFLayer, LIFState
from research.rstdp_spiking.plasticity import STDPWindow, build_trace
from research.rstdp_spiking.spike_encoding import spike_encode

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
        trace_rule: str = "pair-stdp",
        weight_norm: float = 1.0,
        normalize: bool = True,
        action_sampling: str = "boltzmann",
        readout: str = "count",
        readout_decay: float = 0.9,
        logit_scale: float = 1.0,
        arousal: ArousalChannel | None = None,
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
        if readout not in ("count", "rate"):
            raise ValueError(f"readout 只能是 'count' 或 'rate'，收到 {readout!r}。")
        if not 0.0 < readout_decay < 1.0:
            raise ValueError(f"readout_decay 应在 (0, 1)，收到 {readout_decay!r}。")

        device = device or torch.device("cpu")
        self.n_features = int(n_features)
        self.n_actions = int(n_actions)
        self.steps = int(steps)
        self.window = window or STDPWindow()
        self.learning_rate = float(learning_rate)
        self.weight_norm = float(weight_norm)
        self.normalize = bool(normalize)
        self.action_sampling = action_sampling
        self.readout = readout
        self.readout_decay = float(readout_decay)
        self.logit_scale = float(logit_scale)
        # 唤醒通道：**只调采样温度**，不碰权重。None 时行为与从前逐位相同。
        self.arousal = arousal
        self.device = device
        self.last_entropy = 0.0
        self._entropy_sum = 0.0
        self._entropy_count = 0

        self.layer = LIFLayer(
            n_features,
            n_actions,
            tau=tau,
            threshold=threshold,
            n_refractory=n_refractory,
            generator=generator,
        ).to(device)
        # 规则换的是对象，不是 step/apply 里的 if：默认仍是历史的有符号 pair-stdp。
        self.trace = build_trace(
            trace_rule, n_features, n_actions, decay=trace_decay, device=device
        )
        self.trace_rule = trace_rule
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
    def window_spikes(
        self, features: torch.Tensor, *, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """一个决策窗口的 ``(pre, post)`` 脉冲。

        这是 ``step`` 的输入侧，也是信用分配诊断（A1）量「痕迹该往哪边走」的对象；
        **把它做成对外接口**，诊断就不必伸手进私有方法，改窗口实现时两边不会漂移。
        """
        pre_spikes = self._window(features, generator)
        return pre_spikes, self._simulate(pre_spikes)

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

    def sampling_scale(self) -> float:
        """采样尺度 = logit_scale ÷ 唤醒增益。

        语义写死：**唤醒越高 → 尺度越小 → 分布越平 → 探索越多**。
        ``arousal is None`` 时返回 ``logit_scale``，与从前逐位相同。
        """
        if self.arousal is None:
            return self.logit_scale
        return self.logit_scale / self.arousal.gain(1.0)

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
        pre_spikes, post_spikes = self.window_spikes(features, generator=generator)
        scores = self._policy_scores(post_spikes)
        policy = torch.softmax(self.sampling_scale() * scores, dim=-1)

        # 先把动作定下来，再累计痕迹：符号一致的规则需要知道**被选的是哪一列**。
        # pair-stdp 不看 action，所以这个顺序对默认路径逐位不变（它不消耗随机流）。
        if greedy:
            chosen = int(scores.argmax().item())
        elif self.action_sampling == "boltzmann":
            # 采样在生成器所在的设备上做（CPU 生成器不能直接采 CUDA 张量）；
            # CPU 上逐位不变，CUDA 上把概率搬到 CPU 再采，随机流仍由同一个种子决定。
            probabilities = policy.to(generator.device)
            # 诊断 ②：Boltzmann 动作分布的熵（nats）。只记采样分支，贪婪/ε 分支不计。
            entropy = float(-(probabilities * probabilities.clamp(min=1e-12).log()).sum())
            self.last_entropy = entropy
            self._entropy_sum += entropy
            self._entropy_count += 1
            chosen = int(torch.multinomial(probabilities, 1, generator=generator).item())
        elif exploration > 0 and float(torch.rand(1, generator=generator).item()) < exploration:
            chosen = int(torch.randint(self.n_actions, (1,), generator=generator).item())
        else:
            chosen = int(scores.argmax().item())

        if learn_trace:
            if self.trace_rule == "count-aligned":
                self.trace.accumulate_window(
                    pre_spikes,
                    post_spikes,
                    self.window,
                    chosen,
                    features=features,
                    policy=policy,
                )
            else:
                self.trace.accumulate_window(
                    pre_spikes, post_spikes, self.window, chosen, features=features
                )
        return chosen

    @torch.no_grad()
    def _policy_scores(self, post_spikes: torch.Tensor) -> torch.Tensor:
        """策略读出：``count``＝窗口内硬脉冲计数；``rate``＝窗口末的低通发放率。

        D3 实验只换这一处（见 ``sweep_results/rstdp_spiking_rate_readout_prereg.md``）。
        默认 ``count`` 与历史行为逐位相同。
        """
        if self.readout == "count":
            return post_spikes.sum(dim=0).squeeze(0)
        running = torch.zeros(post_spikes.shape[-1], device=post_spikes.device)
        for t in range(post_spikes.shape[0]):
            running = self.readout_decay * running + post_spikes[t, 0]
        return running

    @torch.no_grad()
    def apply_success_signal(self, success_signal: float) -> None:
        """把成功信号写进权重，然后归一化；具体规则由痕迹对象决定。

        - ``pair-stdp``（默认）：``w ← w + η·S·e``；
        - ``sign-consistent``：``S > 0`` 只加 ``η·S·E⁺``、``S < 0`` 只减 ``η·|S|·E⁻``。

        痕迹的清理由 :meth:`end_episode` 负责。
        """
        self.trace.apply(self.layer.weight, self.learning_rate, float(success_signal))
        if self.normalize:
            self._normalize()

    @property
    def episode_entropy(self) -> float:
        """本回合采样决策的平均熵（nats）；没有采样决策时为 0。**读完后由 end_episode 清零。**"""
        return self._entropy_sum / self._entropy_count if self._entropy_count else 0.0

    def state_dict(self) -> dict[str, torch.Tensor]:
        """权重快照；与 :meth:`load_state_dict` 配对，供训练循环选最好一步。"""
        return self.layer.state_dict()

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.layer.load_state_dict(state)

    @torch.no_grad()
    def end_episode(self) -> None:
        self.trace.reset()
        self._entropy_sum = 0.0
        self._entropy_count = 0

    @torch.no_grad()
    def trace_rms(self) -> float:
        return self.trace.rms()
