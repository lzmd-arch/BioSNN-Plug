"""阶段五 M3i：STDP 侧向传播的自主触发机制（计划书任务 3，协议层）。

机制（与预注册一致，纯 stdlib）：

* ``N`` 个 LIF 触发神经元：``v ← α·v + I − thr·z + 侧向抑制``，``α = exp(−1/τ)``；
* 输入 ``I_i = 脉冲_i·w_in + bias + ξ_i``，``脉冲_i ~ Bernoulli(salience)``，
  ``ξ_i ~ N(0, σ_bg²)``（背景噪声维持低发放）；
* **侧向 STDP**：同一群体内，前一步发放的 ``i`` 与当前步发放的 ``j`` 构成因果对，
  ``W[i][j] += η``（增强抑制）；反因果对 ``W[i][j] -= η``；权重截断到 ``[-1, 0]``；
  当前步发放的神经元对其它神经元的膜电位施加 ``+W[i][j]``（负值即抑制）；
* **冲动检测**：群体发放率 ``r_t`` 超过慢均值 ``μ_t + k·σ_t``（EMA，τ=200 ms）；
* **显著性阈值**：冲动连续 ``n_hits`` 步才触发；触发后冷却 ``cooldown`` 步。

只做机制的最小复现：不跨模块、不接真实对话。
"""

from __future__ import annotations

import math
import random

__all__ = ["SpikingTrigger"]


class SpikingTrigger:
    """LIF 群体 + 侧向 STDP + 噪声背景的触发检测器。

    Args:
        n_neurons: 群体大小（≥2）。
        tau: 膜时间常数（ms，>0）。
        threshold: 发放阈值（>0）。
        bias: 常数背景电流。
        noise_sigma: 每步噪声标准差（≥0）。
        input_weight: 输入脉冲权重。
        lateral_eta: 侧向 STDP 步长（≥0）。
        ema_tau: 发放率 EMA 的时间常数（ms，>0）。
        k: 冲动阈值（均值 + k·标准差）。
        n_hits: 连续冲动达到多少步才触发（≥1）。
        cooldown: 触发后的冷却步数（≥0）。
        seed: 随机种子。

    Raises:
        ValueError: 参数越界。
    """

    def __init__(
        self,
        n_neurons: int = 64,
        *,
        tau: float = 20.0,
        threshold: float = 1.0,
        bias: float = 0.02,
        noise_sigma: float = 0.05,
        input_weight: float = 1.0,
        lateral_eta: float = 0.01,
        ema_tau: float = 200.0,
        k: float = 3.0,
        n_hits: int = 5,
        hit_window: int = 1,
        cooldown: int = 200,
        seed: int = 0,
    ) -> None:
        if not isinstance(n_neurons, int) or isinstance(n_neurons, bool) or n_neurons < 2:
            raise ValueError(f"n_neurons 必须是 ≥2 的整数，收到 {n_neurons!r}。")
        if tau <= 0 or ema_tau <= 0:
            raise ValueError("tau 与 ema_tau 必须为正。")
        if threshold <= 0:
            raise ValueError(f"threshold 必须为正，收到 {threshold!r}。")
        if noise_sigma < 0 or lateral_eta < 0:
            raise ValueError("noise_sigma 与 lateral_eta 必须非负。")
        if not isinstance(n_hits, int) or isinstance(n_hits, bool) or n_hits < 1:
            raise ValueError(f"n_hits 必须是 ≥1 的整数，收到 {n_hits!r}。")
        if not isinstance(hit_window, int) or isinstance(hit_window, bool) or hit_window < 1:
            raise ValueError(f"hit_window 必须是 ≥1 的整数，收到 {hit_window!r}。")
        if not isinstance(cooldown, int) or isinstance(cooldown, bool) or cooldown < 0:
            raise ValueError(f"cooldown 必须是非负整数，收到 {cooldown!r}。")

        self.n_neurons = n_neurons
        self.alpha = math.exp(-1.0 / tau)
        self.threshold = float(threshold)
        self.bias = float(bias)
        self.noise_sigma = float(noise_sigma)
        self.input_weight = float(input_weight)
        self.lateral_eta = float(lateral_eta)
        self.ema_decay = math.exp(-1.0 / ema_tau)
        self.k = float(k)
        self.n_hits = n_hits
        # hit_window=1 表示「连续 n_hits 步」（M3i 的判定）；>1 表示滑窗累计（M3j）。
        self.hit_window = hit_window
        self.cooldown = cooldown
        self._rng = random.Random(seed)

        self._v = [0.0] * n_neurons
        self._z_prev = [0] * n_neurons
        self._lateral = [[0.0] * n_neurons for _ in range(n_neurons)]
        self._rate_mean = 0.0
        self._rate_var = 0.0
        self._hits = 0
        self._impulse_history: list[int] = []
        self._cooldown_left = 0

    def _step_inputs(self, salience: float) -> list[float]:
        inputs = []
        for _ in range(self.n_neurons):
            spike = 1.0 if self._rng.random() < salience else 0.0
            noise = self._rng.gauss(0.0, self.noise_sigma)
            inputs.append(spike * self.input_weight + self.bias + noise)
        return inputs

    def process(self, salience: float) -> bool:
        """推进 1 ms；返回本步是否触发。"""
        if not 0.0 <= salience <= 1.0:
            raise ValueError(f"salience 应在 [0, 1]，收到 {salience!r}。")
        inputs = self._step_inputs(salience)
        spikes = [1 if self._v[index] + inputs[index] >= self.threshold else 0
                  for index in range(self.n_neurons)]

        # 侧向 STDP：因果对（z_prev[i] 且 z[j]）增强抑制，反因果对减弱。
        firing_prev = [index for index, value in enumerate(self._z_prev) if value]
        firing_now = [index for index, value in enumerate(spikes) if value]
        for pre in firing_prev:
            for post in firing_now:
                if pre == post:
                    continue
                value = self._lateral[pre][post] + self.lateral_eta
                self._lateral[pre][post] = max(-1.0, min(0.0, value))
        for pre in firing_now:
            for post in firing_prev:
                if pre == post:
                    continue
                value = self._lateral[pre][post] - self.lateral_eta
                self._lateral[pre][post] = max(-1.0, min(0.0, value))

        # 侧向抑制电流：当前发放的神经元压低其它神经元。
        lateral_current = [0.0] * self.n_neurons
        for pre in firing_now:
            row = self._lateral[pre]
            for post in range(self.n_neurons):
                if row[post]:
                    lateral_current[post] += row[post]

        new_v = []
        for index in range(self.n_neurons):
            reset = self.threshold * self._z_prev[index]
            value = self.alpha * self._v[index] + inputs[index] - reset
            value += lateral_current[index]
            new_v.append(value)

        self._v = new_v
        self._z_prev = spikes
        rate = sum(spikes) / self.n_neurons
        delta = rate - self._rate_mean
        self._rate_mean = self.ema_decay * self._rate_mean + (1 - self.ema_decay) * rate
        self._rate_var = self.ema_decay * self._rate_var + (1 - self.ema_decay) * delta * delta
        impulse = rate > self._rate_mean + self.k * math.sqrt(self._rate_var + 1e-9)

        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            self._hits = 0
            self._impulse_history.clear()
            return False
        if self.hit_window == 1:
            self._hits = self._hits + 1 if impulse else 0
        else:
            self._impulse_history.append(1 if impulse else 0)
            if len(self._impulse_history) > self.hit_window:
                self._impulse_history.pop(0)
            self._hits = sum(self._impulse_history)
        if self._hits >= self.n_hits:
            self._hits = 0
            self._impulse_history.clear()
            self._cooldown_left = self.cooldown
            return True
        return False
