"""e-prop 学习器：三因子更新。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

    ΔW_ij = −η · Σ_t L_j^t · e_ij^t

``e_ij^t`` 是资格痕迹（:mod:`research.eprop.traces`，局部），``L_j^t`` 是学习信号
（来自输出误差，全局但只是一个每神经元每时刻的标量）。

## 学习路径上没有任何 autograd

这是本模块与"用 BPTT 训练"最实质的区别，也是计划书 §9「无全局反向传播」的落点：

* 前向模拟**不建计算图**（``torch.no_grad``），脉冲网络内部没有可回传的图；
* 资格痕迹是显式递推出来的，不是求导得到的；
* 学习信号有**闭式**：``dE/dz_filt = W_out @ dE/dy``，其中 ``dE/dy = softmax(ŷ) −
  onehot``（本项目只做分类任务，所以只实现了交叉熵这一支）。一行矩阵乘法，不需要 autograd；
* 读出层用普通的梯度下降更新（它本来就是个线性分类器，这是论文的做法）。

官方实现里学习信号是 ``tf.gradients(loss, filtered_z)`` ——求导只到**滤波后的脉冲**为止，
不会进入脉冲网络。本模块把这一步也换成闭式，于是"局部"这件事在代码里一眼可查：
整个文件没有一次 ``backward()`` 或 ``autograd.grad``。

## 损失取在哪些时刻

默认只在**最后一个时间步**算损失。理由：a 序列的前几步还没看完整输入，在那里要求分类
只会引入一个不可学的噪声底。而资格痕迹在最后一步已经累积了整段历史，所以更新仍然用到了
全序列的信息——这正是 e-prop 用痕迹换掉 BPTT 的地方。

``loss_timesteps="all"`` 可以改成逐步都算，用于对照。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from research.eprop.neurons import ALIFCell, ALIFState
from research.eprop.traces import eligibility_traces, exp_convolve

__all__ = ["EPropLearner", "Rollout"]


@dataclass
class Rollout:
    """一次前向模拟的产物，全部**不含计算图**。"""

    spikes: torch.Tensor  # (T, batch, n_rec)
    v_scaled: torch.Tensor  # (T, batch, n_rec)
    filtered: torch.Tensor  # (T, batch, n_rec)，指数滤波后的脉冲，读出的输入


class EPropLearner(torch.nn.Module):
    """一层 ALIF 循环网络 + 线性读出。

    隐藏层（``w_in``、``w_rec``）由 e-prop 更新；读出（``w_out``）由梯度下降更新。

    Attributes:
        cell: 循环层，持有 ``w_in`` 与 ``w_rec``。
        w_out: ``(n_rec, n_out)`` 线性读出权重。
    """

    def __init__(
        self,
        n_in: int,
        n_rec: int,
        n_out: int,
        *,
        tau: float = 20.0,
        tau_adaptation: float = 500.0,
        threshold: float = 0.62,
        beta: float = 0.07,
        dampening_factor: float = 0.3,
        n_refractory: int = 2,
        learning_rate_in: float = 1e-3,
        learning_rate_rec: float = 1e-3,
        learning_rate_out: float = 1e-2,
        decay_out: float = 0.95,
        loss_timesteps: str = "last",
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if loss_timesteps not in ("last", "all"):
            raise ValueError(f"loss_timesteps 只能是 'last' 或 'all'，收到 {loss_timesteps!r}。")

        self.cell = ALIFCell(
            n_in,
            n_rec,
            tau=tau,
            tau_adaptation=tau_adaptation,
            threshold=threshold,
            beta=beta,
            dampening_factor=dampening_factor,
            n_refractory=n_refractory,
            generator=generator,
        )
        self.n_in = int(n_in)
        self.n_rec = int(n_rec)
        self.n_out = int(n_out)

        out_generator = generator or torch.Generator()
        self.w_out = torch.nn.Parameter(
            torch.randn(n_rec, n_out, generator=out_generator) / (n_rec**0.5)
        )

        self.learning_rate_in = float(learning_rate_in)
        self.learning_rate_rec = float(learning_rate_rec)
        self.learning_rate_out = float(learning_rate_out)
        self.decay_out = float(decay_out)
        self.loss_timesteps = loss_timesteps

    # ---------------------------------------------------------------- 前向

    @torch.no_grad()
    def simulate(self, inputs: torch.Tensor) -> Rollout:
        """跑完整序列，返回脉冲与缩放后的膜电位。**全程无计算图。**"""
        steps, batch = inputs.shape[0], inputs.shape[1]
        state = ALIFState.zeros(batch, self.n_rec, device=inputs.device, dtype=inputs.dtype)
        spikes, scaled = [], []
        for t in range(steps):
            z, state = self.cell.step(inputs[t], state)
            spikes.append(z)
            scaled.append(self.cell.v_scaled(state.v, state.a))
        stacked = torch.stack(spikes)
        return Rollout(
            spikes=stacked,
            v_scaled=torch.stack(scaled),
            filtered=exp_convolve(stacked, self.decay_out),
        )

    @torch.no_grad()
    def logits(self, rollout: Rollout) -> torch.Tensor:
        """读出输出，``(T, batch, n_out)``。"""
        return rollout.filtered @ self.w_out

    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """最终时刻的分类结果，``(batch,)``。"""
        return self.logits(self.simulate(inputs))[-1].argmax(dim=-1)

    # ---------------------------------------------------------------- 损失

    @torch.no_grad()
    def _selected_logits(self, rollout: Rollout) -> torch.Tensor:
        """参与损失的读出输出，``(T', batch, n_out)``。"""
        filtered = rollout.filtered if self.loss_timesteps == "all" else rollout.filtered[-1:]
        return filtered @ self.w_out

    @torch.no_grad()
    def loss_value(self, rollout: Rollout, targets: torch.Tensor) -> float:
        """分类交叉熵，用于记录曲线。``targets`` 是 ``(batch,)`` 的类别下标。"""
        logits = self._selected_logits(rollout)
        steps, batch, n_out = logits.shape
        # 同一段序列里每个时刻用同一个标签（"all" 模式下也如此）
        repeated = targets.reshape(1, batch).expand(steps, batch).reshape(-1)
        return float(torch.nn.functional.cross_entropy(logits.reshape(-1, n_out), repeated).item())

    @torch.no_grad()
    def output_gradient(self, rollout: Rollout, targets: torch.Tensor) -> torch.Tensor:
        """``dE/dy = (softmax(ŷ) − onehot) / (T'·batch)``，``(T', batch, n_out)``。"""
        logits = self._selected_logits(rollout)
        steps, batch, _ = logits.shape
        probabilities = torch.softmax(logits, dim=-1)
        one_hot = torch.zeros_like(probabilities)
        repeated = targets.reshape(1, batch, 1).expand(steps, batch, 1)
        one_hot.scatter_(-1, repeated, 1.0)
        return (probabilities - one_hot) / (steps * batch)

    @torch.no_grad()
    def learning_signal(self, rollout: Rollout, targets: torch.Tensor) -> torch.Tensor:
        """``L_j^t = (W_out · dE/dy)_j^t``，``(T, batch, n_rec)``。

        只在有损失的时刻非零，其余时刻为 0——所以"损失只取最后一步"时，整个序列里只有
        最后一步的学习信号非零，而痕迹在该步已累积了整段历史。

        ``dE/dz_filt = W_out @ dE/dy`` 是线性读出的**闭式**梯度，不用 autograd：
        ``z_filt`` 与 ``y`` 之间只隔一个 ``W_out``。
        """
        gradient = self.output_gradient(rollout, targets) @ self.w_out.transpose(0, 1)
        signal = torch.zeros_like(rollout.spikes)
        if self.loss_timesteps == "all":
            signal = gradient
        else:
            signal[-1:] = gradient
        return signal

    # ---------------------------------------------------------------- 更新

    @torch.no_grad()
    def update(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """一步在线更新：模拟 → 学习信号 → 痕迹 → 更新三个权重矩阵。

        Returns:
            更新前的损失值。
        """
        rollout = self.simulate(inputs)
        loss = self.loss_value(rollout, targets)
        signal = self.learning_signal(rollout, targets)

        # 前突触脉冲：输入直接是 x^t；循环连接用的是上一步的脉冲（官方实现传 z_previous_step）。
        z_prev = torch.cat([torch.zeros_like(rollout.spikes[:1]), rollout.spikes[:-1]])

        for z_pre, weight, rate, recurrent in (
            (inputs, self.cell.w_in, self.learning_rate_in, False),
            (z_prev, self.cell.w_rec, self.learning_rate_rec, True),
        ):
            trace = eligibility_traces(
                rollout.v_scaled,
                z_pre,
                rollout.spikes,
                alpha=self.cell.alpha,
                rho=self.cell.rho,
                beta=self.cell.beta,
                threshold=self.cell.threshold,
                dampening_factor=self.cell.dampening_factor,
                n_refractory=self.cell.n_refractory,
                is_recurrent=recurrent,
            )
            gradient = torch.einsum("btj,btij->ij", signal, trace)
            weight.add_(gradient, alpha=-rate)

        # 读出：普通的梯度下降（它是线性分类器，这是论文的做法）
        # dE/dW_out = Σ_{t,b} z_filt[t,b,:]^T · dE/dy[t,b,:]
        filtered = rollout.filtered if self.loss_timesteps == "all" else rollout.filtered[-1:]
        self.w_out.add_(
            -self.learning_rate_out
            * torch.einsum("tbi,tbo->io", filtered, self.output_gradient(rollout, targets))
        )
        return loss

    # ---------------------------------------------------------------- 指标

    @torch.no_grad()
    def active_neuron_fraction(self, inputs: torch.Tensor) -> float:
        """计划书 §9 的**活跃神经元比例**：整个评估窗内至少发放过一次的神经元占比。

        这是**原义**指标（脉冲网络），不是 W1 里那种速率型类比量：判据就是脉冲张量里
        该神经元是否出现过非零值。§3.1 规定低于 0.60 触发阈值调整，§七 第一阶段要求 > 60%。
        """
        spikes = self.simulate(inputs).spikes  # (T, batch, n_rec)
        # 必须是 (T·batch, n_rec)：把 batch 与 n_rec 两个轴混在一起数，得到的是
        # "（样本, 神经元）对里有多大比例发放过"，那是另一个量，而且数值偏小。
        fired = spikes.reshape(-1, self.n_rec) > 0
        return float(fired.any(dim=0).float().mean().item())

    @torch.no_grad()
    def spike_sparsity(self, inputs: torch.Tensor) -> float:
        """计划书 §9 的脉冲稀疏度：零元素占比。"""
        spikes = self.simulate(inputs).spikes
        return float((spikes == 0).float().mean().item())
