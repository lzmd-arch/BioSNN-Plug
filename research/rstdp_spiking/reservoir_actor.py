"""带隐藏群体的脉冲执行层（v3 预注册的 H1/H2 两臂）。

结构：输入 → **冻结的随机隐藏层** → 线性读出 → 动作。

- **H1**：读出权重由**两因子局部规则**更新（奖励 × 前后活动），不经过隐藏层、不用反向传播；
- **H2**：读出权重由岭回归一次拟合（**监督读出，越红线**，只作上界参照）。

冻结隐藏层这件事是刻意的：它把"群体在不在"和"读出来怎么学"拆成两个变量。
如果连冻结的随机群体都不行，说明问题不在读出；如果 H2 行而 H1 不行，
说明瓶颈在读出的学习方式——这两种结论都只有在隐藏层冻结时才干净。
"""

from __future__ import annotations

import torch

from .neurons import LIFLayer, LIFState
from .population_readout import RidgeReadout

__all__ = ["ReservoirActor"]


class ReservoirActor:
    """冻结储层 + 可学读出。

    Args:
        n_features: 输入维数。
        n_hidden: 隐藏群体规模。v3 预注册固定为 128。
        n_actions: 动作数。
        steps: 一个决策窗口的时间步数。
        seed: 隐藏层权重的初始化种子（**冻结前**唯一影响它的东西）。
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        n_actions: int,
        *,
        steps: int = 25,
        seed: int = 0,
        device: torch.device | None = None,
        learning_rate: float = 1e-2,
    ) -> None:
        if n_hidden <= n_actions:
            raise ValueError(
                f"隐藏群体必须明显大于动作数，收到 n_hidden={n_hidden}、n_actions={n_actions}。"
            )
        self.n_features = int(n_features)
        self.n_hidden = int(n_hidden)
        self.n_actions = int(n_actions)
        self.steps = int(steps)
        self.device = device or torch.device("cpu")
        self.learning_rate = float(learning_rate)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.hidden = LIFLayer(n_features, n_hidden).to(self.device)
        with torch.no_grad():
            noise = torch.randn(
                self.hidden.weight.shape, generator=generator, dtype=self.hidden.weight.dtype
            )
            self.hidden.weight.copy_(noise / (n_features**0.5))
        for parameter in self.hidden.parameters():
            parameter.requires_grad_(False)

        self.w_out = torch.zeros(n_hidden, n_actions, device=self.device)
        self.decoder: RidgeReadout | None = None

    # ---------------------------------------------------------------- 前向
    @torch.no_grad()
    def hidden_spikes(self, features: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
        """跑一个窗口，返回隐藏层脉冲 ``(T, 1, n_hidden)``。"""
        from .spike_encoding import spike_encode

        pre = spike_encode(features, self.steps, generator=generator).unsqueeze(1)
        state = LIFState.zeros(1, self.n_hidden, device=self.device)
        collected = []
        for t in range(self.steps):
            spikes, state = self.hidden.step(pre[t], state)
            collected.append(spikes)
        return torch.stack(collected)

    @torch.no_grad()
    def counts(self, spikes: torch.Tensor) -> torch.Tensor:
        """整窗计数 ``(n_hidden,)``——这是"群体向量"。"""
        return spikes.sum(dim=0).squeeze(0)

    @torch.no_grad()
    def scores(self, counts: torch.Tensor) -> torch.Tensor:
        if self.decoder is not None:
            return self.decoder.scores(counts.unsqueeze(0)).squeeze(0)
        return counts @ self.w_out

    @torch.no_grad()
    def step(
        self, features: torch.Tensor, *, generator: torch.Generator, greedy: bool = False
    ) -> int:
        counts = self.counts(self.hidden_spikes(features, generator=generator))
        logits = self.scores(counts)
        if greedy:
            return int(logits.argmax().item())
        probabilities = torch.softmax(logits, dim=-1).to(generator.device)
        return int(torch.multinomial(probabilities, 1, generator=generator).item())

    # ---------------------------------------------------------------- H1：局部规则
    @torch.no_grad()
    def reinforce(self, counts: torch.Tensor, action: int, reward: float) -> None:
        """两因子局部更新：``w[:, a] += η · reward · (counts − 均值)``。

        **不碰隐藏层**，也不需要任何上游梯度——这正是 H1 与 H2 的分界。
        """
        centred = counts - counts.mean()
        self.w_out[:, action] += self.learning_rate * float(reward) * centred

    # ---------------------------------------------------------------- H2：监督读出
    def fit_decoder(
        self, counts: torch.Tensor, actions: torch.Tensor, *, alpha: float = 1.0
    ) -> None:
        """用岭回归拟合读出（H2）。``actions`` 是 ``(n,)`` 的动作下标。"""
        targets = torch.zeros(counts.shape[0], self.n_actions, device=self.device)
        targets[torch.arange(counts.shape[0]), actions] = 1.0
        self.decoder = RidgeReadout(alpha=alpha).fit(counts, targets)
