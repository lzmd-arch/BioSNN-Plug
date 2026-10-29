"""Predictive E-prop 最小实现：用**局部预测误差**当第三因子。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

对应 [Noè et al., 2026] 的 Predictive E-prop。原文的核心主张是：把 e-prop 的第三因子
从「任务专用的外部信号」换成**预测编码自身的局部预测误差**，网络只靠自己的预测误差做
自监督学习；学成后可以在关掉误差输入（自由运行）时自行维持目标动力学。

## 本实现与原文的关系（先声明，不冒充复现）

原文（bioRxiv 2026.02.12.705507）的公式在网页上是图片，**本轮没有逐字取得方程**。
本模块按正文描述实现：

* 循环层：LIF/ALIF（复用 :class:`~research.eprop.neurons.ALIFCell`），
  ``beta=0`` 即全 LIF；
* 固定输入权重：预测误差 ``e_t = x_t − ŷ_t`` 经固定随机 ``W_in`` 回注，自身输出 ``ŷ_t``
  经固定随机反馈矩阵回注（原文 "to stabilize neural dynamics"）；
* 读出：脉冲经**双指数滤波**后乘 ``W_out`` 得到预测 ``ŷ_t``；
* 第三因子：``L_t = (−e_t)·B``，``B`` 是固定随机矩阵（广播对齐，Lillicrap et al.）；
* 更新：循环权重走 e-prop 资格痕迹（:func:`~research.eprop.traces.eprop_gradient`），
  读出走普通梯度下降；可选发放率稳态项与权重衰减。

**已知差异**：更新在 trial 末执行（原文每 ``t_delay`` ms 更新一次）；没有 OU 噪声项；
没有混用 LIF/ALIF（整层同 β）；没有 TBPTT 基线。这些差异写进结果 README，
不得把本实现的数字当成原文结论的复现。

## 学习路径上没有 autograd

前向全程 ``torch.no_grad``；没有 ``backward()``，也没有 ``nn.Parameter``。循环权重的
更新来自显式资格痕迹，读出的更新来自闭式梯度。这一点与
:mod:`research.eprop.eprop` 相同，是「无全局反向传播」的落点。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from research.eprop.neurons import ALIFCell, ALIFState
from research.eprop.traces import eprop_gradient

__all__ = ["PredictiveEProp", "PredictiveRollout"]


@dataclass
class PredictiveRollout:
    """一次前向的产物，全部不含计算图。"""

    spikes: torch.Tensor  # (T, batch, n_rec)
    v_scaled: torch.Tensor  # (T, batch, n_rec)
    filtered: torch.Tensor  # (T, batch, n_rec)，双指数滤波后的脉冲
    prediction: torch.Tensor  # (T, batch, n_signal)
    error: torch.Tensor  # (T, batch, n_signal)，target − prediction


class PredictiveEProp:
    """预测编码式脉冲循环网络 + 线性读出。

    Args:
        n_signal: 信号维度（正弦任务为 1）。
        n_rec: 循环神经元数。
        gain: 固定输入权重与反馈权重的增益（原文的 ``g``）。
        beta: ALIF 适应强度；0 即 LIF。
        tau_rise / tau_decay: 读出的双指数滤波时间常数（ms）。
        learning_rate_rec / learning_rate_out: 循环/读出学习率。
        lambda_reg: 发放率稳态项强度（目标 10 Hz）。
        lambda_weight: 读出权重衰减。
        generator: 随机数生成器，保证可复现。
    """

    def __init__(
        self,
        n_signal: int = 1,
        n_rec: int = 300,
        *,
        gain: float = 1.0,
        beta: float = 0.07,
        tau: float = 20.0,
        tau_adaptation: float = 500.0,
        threshold: float = 0.62,
        dampening_factor: float = 0.3,
        n_refractory: int = 2,
        tau_rise: float = 5.0,
        tau_decay: float = 20.0,
        learning_rate_rec: float = 1e-4,
        learning_rate_out: float = 1e-2,
        lambda_reg: float = 1e-3,
        lambda_weight: float = 1e-5,
        target_rate: float = 0.01,
        bias_scale: float = 0.1,
        generator: torch.Generator | None = None,
    ) -> None:
        if n_signal < 1 or n_rec < 1:
            raise ValueError(f"n_signal、n_rec 必须为正，收到 {n_signal}、{n_rec}。")
        self.n_signal = int(n_signal)
        self.n_rec = int(n_rec)
        self.gen = generator or torch.Generator()

        n_in = 2 * n_signal + 1  # 误差通道 + 自身输出反馈通道 + 偏置
        self.cell = ALIFCell(
            n_in,
            n_rec,
            tau=tau,
            tau_adaptation=tau_adaptation,
            threshold=threshold,
            beta=beta,
            dampening_factor=dampening_factor,
            n_refractory=n_refractory,
            generator=self.gen,
        )
        # 固定输入权重：前 n_signal 行接预测误差，接着 n_signal 行接自身输出，最后一行是偏置。
        fixed = torch.zeros(n_in, n_rec)
        fixed[: 2 * n_signal] = torch.randn(2 * n_signal, n_rec, generator=self.gen) * (
            gain / n_signal**0.5
        )
        fixed[-1] = torch.randn(n_rec, generator=self.gen) * bias_scale
        with torch.no_grad():
            self.cell.w_in.copy_(fixed)

        self.w_out = torch.randn(n_rec, n_signal, generator=self.gen) / n_rec**0.5
        # 广播对齐矩阵：把预测误差映射成每神经元的第三因子。
        self.feedback = torch.randn(n_signal, n_rec, generator=self.gen) / n_signal**0.5

        self.learning_rate_rec = float(learning_rate_rec)
        self.learning_rate_out = float(learning_rate_out)
        self.lambda_reg = float(lambda_reg)
        self.lambda_weight = float(lambda_weight)
        self.target_rate = float(target_rate)
        self.a_rise = float(torch.exp(torch.tensor(-1.0 / tau_rise)))
        self.a_decay = float(torch.exp(torch.tensor(-1.0 / tau_decay)))

    # ---------------------------------------------------------------- 前向

    @torch.no_grad()
    def simulate(
        self,
        target: torch.Tensor | None = None,
        *,
        steps: int | None = None,
        error_gain: float = 1.0,
    ) -> PredictiveRollout:
        """跑一段序列。``target`` 为空时是自由运行（误差输入为 0）。

        Args:
            target: ``(T, batch, n_signal)`` 的目标信号；None 表示自由运行。
            steps: 自由运行时的步数。
            error_gain: 误差输入的开关系数 ``ξ``（论文用 0/1；训练时可小于 1 作平滑）。
        """
        if target is None:
            if steps is None:
                raise ValueError("自由运行时必须给 steps。")
            t_steps, batch = int(steps), 1
            dtype = self.w_out.dtype
        else:
            if target.dim() != 3 or target.shape[-1] != self.n_signal:
                raise ValueError(
                    f"target 必须是 (T, batch, {self.n_signal})，收到 {tuple(target.shape)}。"
                )
            t_steps, batch = target.shape[0], target.shape[1]
            dtype = target.dtype

        state = ALIFState.zeros(batch, self.n_rec, dtype=dtype)
        pred_prev = torch.zeros(batch, self.n_signal)
        err_prev = torch.zeros(batch, self.n_signal)
        s1 = torch.zeros(batch, self.n_rec)
        s2 = torch.zeros(batch, self.n_rec)
        ones = torch.ones(batch, 1)
        gain = (1.0 - self.a_rise) * (1.0 - self.a_decay)

        spikes, scaled, filtered, predictions, errors = [], [], [], [], []
        for t in range(t_steps):
            inputs = torch.cat([error_gain * err_prev, pred_prev, ones], dim=-1)
            z, state = self.cell.step(inputs, state)
            v_scaled = self.cell.v_scaled(state.v, state.a)
            s1 = self.a_rise * s1 + z
            s2 = self.a_decay * s2 + s1
            filt = s2 * gain
            prediction = filt @ self.w_out
            error = (target[t] - prediction) if target is not None else torch.zeros_like(prediction)

            spikes.append(z)
            scaled.append(v_scaled)
            filtered.append(filt)
            predictions.append(prediction)
            errors.append(error)
            pred_prev, err_prev = prediction, error

        return PredictiveRollout(
            spikes=torch.stack(spikes),
            v_scaled=torch.stack(scaled),
            filtered=torch.stack(filtered),
            prediction=torch.stack(predictions),
            error=torch.stack(errors),
        )

    @torch.no_grad()
    def reconstruction_mse(self, target: torch.Tensor) -> float:
        """误差回注开启时的重建 MSE（训练分布内的自监督指标）。"""
        rollout = self.simulate(target, error_gain=1.0)
        return float((rollout.error**2).mean().item())

    # ---------------------------------------------------------------- 更新

    @torch.no_grad()
    def update(
        self,
        target: torch.Tensor,
        *,
        chunk: int | None = None,
        error_gain: float = 1.0,
    ) -> float:
        """一个 trial 的在线更新。返回更新前的重建 MSE。

        ``chunk`` 为 None 时在 trial 末执行一次权重更新（与
        :func:`~research.eprop.traces.eprop_gradient` 完全一致）；给出 ``chunk`` 时每
        ``chunk`` 个时间步更新一次，**资格痕迹跨更新保持不清零**——这更接近原文「每
        ``t_delay`` ms 更新一次」的做法，也让同样的一段序列产生更多次更新。
        """
        rollout = self.simulate(target, error_gain=error_gain)
        error = rollout.error
        mse = float((error**2).mean().item())

        # 第三因子：广播对齐。dE/dŷ = −e/(T·batch)，再乘固定随机 B。
        signal = torch.einsum("tbd,dn->tbn", -error, self.feedback)  # (T, batch, n_rec)
        if self.lambda_reg:
            rate = rollout.spikes.mean(dim=0)  # (batch, n_rec)，10 Hz ⇔ 0.01 脉冲/ms
            signal = signal - (2.0 * self.lambda_reg * (rate - self.target_rate)).unsqueeze(0)

        z_prev = torch.cat([torch.zeros_like(rollout.spikes[:1]), rollout.spikes[:-1]])
        if chunk is None:
            gradient = eprop_gradient(
                rollout.v_scaled,
                z_prev,
                rollout.spikes,
                signal,
                alpha=self.cell.alpha,
                rho=self.cell.rho,
                beta=self.cell.beta,
                threshold=self.cell.threshold,
                dampening_factor=self.cell.dampening_factor,
                n_refractory=self.cell.n_refractory,
                is_recurrent=True,
            )
            self.cell.w_rec.add_(gradient, alpha=-self.learning_rate_rec)
            self._update_readout(rollout.filtered, error, weight=1.0)
            return mse

        if chunk < 1:
            raise ValueError(f"chunk 必须 >= 1，收到 {chunk}。")
        self._update_chunked(rollout, z_prev, signal, error, chunk)
        return mse

    @torch.no_grad()
    def _update_chunked(
        self,
        rollout: PredictiveRollout,
        z_prev: torch.Tensor,
        signal: torch.Tensor,
        error: torch.Tensor,
        chunk: int,
    ) -> None:
        """分段更新，资格痕迹跨段保持。

        与 :func:`~research.eprop.traces.eprop_gradient` 的递推逐行相同；差别只在
        「何时把累积的梯度落到权重上」。``chunk >= T`` 时与单次更新**逐位相同**，
        见 ``tests/test_predictive.py::test_chunked_equals_single_update_at_full_length``。
        """
        from research.eprop.traces import refractory_mask

        spikes = rollout.spikes
        t_steps, batch, n_post = spikes.shape
        identity = torch.eye(n_post)
        psi_full = (self.cell.dampening_factor / self.cell.threshold) * torch.clamp(
            1.0 - rollout.v_scaled.abs(), min=0.0
        )
        psi = torch.where(
            refractory_mask(spikes, self.cell.n_refractory), torch.zeros_like(psi_full), psi_full
        )

        epsilon_v = z_prev[0][:, :, None].clone()  # (batch, n_pre, 1)
        epsilon_a = torch.zeros(batch, n_post, n_post)
        gradient = torch.zeros(n_post, n_post)
        segment_start = 0
        for t in range(t_steps):
            if t > 0:
                epsilon_v = self.cell.alpha * epsilon_v + z_prev[t][:, :, None]
            psi_t = psi[t][:, None, :]
            epsilon_a = (self.cell.rho - self.cell.beta * psi_t) * epsilon_a + psi_t * epsilon_v
            trace = psi_t * (epsilon_v - self.cell.beta * epsilon_a)
            gradient += torch.einsum("bj,bij->ij", signal[t], trace)
            if (t + 1) % chunk == 0 or t == t_steps - 1:
                gradient = gradient * (1.0 - identity)
                self.cell.w_rec.add_(gradient, alpha=-self.learning_rate_rec)
                gradient = torch.zeros_like(gradient)
                weight = (t + 1 - segment_start) / t_steps
                self._update_readout(
                    rollout.filtered[segment_start : t + 1],
                    error[segment_start : t + 1],
                    weight=weight,
                )
                segment_start = t + 1

    @torch.no_grad()
    def _update_readout(
        self, filtered: torch.Tensor, error: torch.Tensor, *, weight: float
    ) -> None:
        steps, batch, _ = filtered.shape
        grad_out = torch.einsum("tbi,tbd->id", filtered, -error) / (steps * batch)
        self.w_out.add_(grad_out, alpha=-self.learning_rate_out * weight)
        if self.lambda_weight:
            self.w_out.mul_(1.0 - self.learning_rate_out * self.lambda_weight * weight)

    # ---------------------------------------------------------------- 指标

    @torch.no_grad()
    def firing_rate(self, target: torch.Tensor) -> float:
        """平均发放率（Hz，dt = 1 ms）。"""
        return float(self.simulate(target).spikes.mean().item()) * 1000.0
