"""TokenEProp：**逐时刻**的局部学习语言模型（无反向传播）。

与 :class:`research.eprop.EventDrivenEProp` 的差别只有三处（判据见 `research/sel/prereg-lm-eprop.md`）：

1. **逐时刻损失**：每个位置都预测下一个 token，而不是整段一个标签；
2. **streaming**：v / z / 不应期 / 输入痕迹 / 循环痕迹 / 读出滤波**跨块结转**，
   只在会话边界重置——这是"多轮"的前提，不是性能优化；
3. **token 输入**：one-hot 走 `w_in` 的 gather，`w_in` 的行就是可局部学习的 embedding。

**没有任何 autograd**：所有更新都在 ``torch.no_grad()`` 下手写，`step`/`run` 全程不建计算图。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = ["LMTrace", "TokenEProp"]


@dataclass
class LMTrace:
    """跨块结转的全部状态。只在会话边界调用 :meth:`reset`。

    这张表的每一项都**必须**携带过块边界：把其中任何一个在块首清零，
    都会让模型"忘了上一块"，而这不是性能问题、是能力问题。
    """

    v: torch.Tensor  # (B, N) 膜电位
    z: torch.Tensor  # (B, N) 上一步脉冲
    refractory: torch.Tensor  # (B, N) 不应期计数
    eps_in: torch.Tensor  # (B, V) 输入痕迹（one-hot 的衰减计数）
    eps_rec: torch.Tensor  # (B, N) 循环痕迹
    readout: torch.Tensor  # (B, N) 读出 EMA

    @classmethod
    def zeros(cls, batch: int, n_rec: int, n_vocab: int, *, device=None, dtype=torch.float32):
        def zeros(*shape: int) -> torch.Tensor:
            return torch.zeros(*shape, device=device, dtype=dtype)

        return cls(
            v=zeros(batch, n_rec),
            z=zeros(batch, n_rec),
            refractory=zeros(batch, n_rec),
            eps_in=zeros(batch, n_vocab),
            eps_rec=zeros(batch, n_rec),
            readout=zeros(batch, n_rec),
        )

    def reset(self, rows: torch.Tensor) -> None:
        """把指定行清零（会话边界）。``rows`` 是 bool 掩码或下标张量。"""
        for name in ("v", "z", "refractory", "eps_in", "eps_rec", "readout"):
            getattr(self, name)[rows] = 0.0


class TokenEProp:
    """单区 LIF + 逐时刻 e-prop + EMA 读出。

    Args:
        n_vocab: 词表大小。
        n_rec: 循环神经元数。
        fan_in: 固定稀疏扇入（沿用 ``EventDrivenEProp`` 的口径）。
        tau: 膜时间常数（``alpha = exp(-1/tau)``）。
        tau_read: 读出 EMA 的时间常数。
        input_std: **one-hot 输入的初始化标准差**。必须按发放率标定：
            稳态发放率约 ``|电流| / 阈值``，0.1 对应约 16%。用 ``1/sqrt(n_in)``
            会让 one-hot 输入小到神经元根本不发放。
    """

    def __init__(
        self,
        n_vocab: int,
        n_rec: int = 1024,
        *,
        fan_in: int = 32,
        tau: float = 20.0,
        threshold: float = 0.62,
        dampening_factor: float = 0.3,
        n_refractory: int = 2,
        tau_read: float = 8.0,
        input_std: float = 0.12,
        lr_in: float = 2e-4,
        lr_rec: float = 1e-4,
        lr_out: float = 0.5,
        generator: torch.Generator | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.n_vocab = int(n_vocab)
        self.n_rec = int(n_rec)
        self.fan_in = int(fan_in)
        self.threshold = float(threshold)
        self.dampening_factor = float(dampening_factor)
        self.n_refractory = int(n_refractory)
        self.alpha = float(math.exp(-1.0 / tau))
        self.decay_read = float(math.exp(-1.0 / tau_read))
        self.lr_in = float(lr_in)
        self.lr_rec = float(lr_rec)
        self.lr_out = float(lr_out)
        self.device = torch.device(device)

        gen = generator or torch.Generator()
        # one-hot 输入：w_in 的行就是每个 token 的"embedding"，由局部规则学习。
        self.w_in = torch.randn(n_vocab, n_rec, generator=gen).to(self.device) * input_std
        self.w_out = torch.randn(n_rec, n_vocab, generator=gen).to(self.device) / math.sqrt(n_rec)
        post = torch.arange(n_rec, device=self.device).unsqueeze(1)
        candidates = torch.randint(0, n_rec - 1, (n_rec, fan_in), generator=gen).to(self.device)
        self.pre_index = candidates + (candidates >= post).long()
        self.w_rec = torch.randn(n_rec, fan_in, generator=gen).to(self.device) / math.sqrt(fan_in)

    # -- 状态 -----------------------------------------------------------------

    def new_trace(self, batch: int) -> LMTrace:
        return LMTrace.zeros(batch, self.n_rec, self.n_vocab, device=self.device)

    # -- 前向与更新 ------------------------------------------------------------

    def _recurrent_current(self, z_previous: torch.Tensor) -> torch.Tensor:
        current = torch.zeros_like(z_previous)
        for slot in range(self.fan_in):
            current += z_previous[:, self.pre_index[:, slot]] * self.w_rec[:, slot]
        return current

    @torch.no_grad()
    def step(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        trace: LMTrace,
        *,
        update: bool = True,
        learn_hidden: bool = True,
        learn_recurrent: bool = True,
    ) -> tuple[torch.Tensor, int]:
        """跑一个时刻。返回 (逐样本 CE, 本步发放数)。``targets`` 里的 -100 不参与损失。"""
        batch = tokens.shape[0]
        dtype = trace.v.dtype
        valid = targets != -100
        active = valid.to(dtype).unsqueeze(1)  # (B, 1)

        z_previous = trace.z
        current = self.w_in[tokens] + self._recurrent_current(z_previous)
        new_v = self.alpha * trace.v + current - z_previous * self.threshold
        scaled = (new_v - self.threshold) / self.threshold
        refractory = trace.refractory > 0
        fresh = (scaled > 0).to(dtype)  # 硬阈值；本类不做反传，不需要伪导数
        new_z = torch.where(refractory, torch.zeros_like(fresh), fresh)
        new_r = torch.clamp(
            trace.refractory + self.n_refractory * new_z - 1.0,
            min=0.0,
            max=float(self.n_refractory),
        )
        # padding 位置冻结状态：不推进膜电位、不发放
        trace.v = torch.where(active > 0, new_v, trace.v)
        trace.z = torch.where(active > 0, new_z, torch.zeros_like(new_z))
        trace.refractory = torch.where(active > 0, new_r, trace.refractory)
        trace.readout = self.decay_read * trace.readout + trace.z

        # 资格痕迹：**用本步之前的脉冲与本步的输入**累积（与 EventDrivenEProp 同口径）
        trace.eps_rec = self.alpha * trace.eps_rec + z_previous
        trace.eps_in = self.alpha * trace.eps_in
        trace.eps_in.scatter_add_(1, tokens.reshape(-1, 1), active.to(trace.eps_in.dtype))

        logits = trace.readout @ self.w_out  # (B, V)
        loss = torch.nn.functional.cross_entropy(
            logits, targets, ignore_index=-100, reduction="none"
        )
        fired = int(trace.z.sum())

        if update and bool(valid.any()):
            probability = torch.softmax(logits, dim=-1)
            one_hot = torch.zeros_like(probability)
            one_hot.scatter_(1, targets.clamp(min=0).reshape(-1, 1), 1.0)
            gradient = (probability - one_hot) * active
            gradient = gradient / max(batch, 1)
            signal = gradient @ self.w_out.T  # (B, N)
            psi = (self.dampening_factor / self.threshold) * torch.clamp(
                1.0 - scaled.abs(), min=0.0
            )
            psi = torch.where(trace.refractory > 0, torch.zeros_like(psi), psi)
            post = signal * psi * active

            if learn_hidden:
                grad_in = torch.einsum("bn,bv->vn", post, trace.eps_in)
                self.w_in.add_(grad_in, alpha=-self.lr_in)
            if learn_recurrent:
                gathered = trace.eps_rec[:, self.pre_index]  # (B, N, fan_in)
                grad_rec = torch.einsum("bnf,bn->nf", gathered, post)
                self.w_rec.add_(grad_rec, alpha=-self.lr_rec)
            # 读出：NLMS（按 ||y||² 归一化，每个类只看自己的误差）
            norm = float(trace.readout.pow(2).sum(dim=1).mean().clamp(min=1e-3))
            self.w_out.add_(trace.readout.T @ gradient / norm, alpha=-self.lr_out)
        return loss, fired

    @torch.no_grad()
    def run(
        self,
        tokens: torch.Tensor,
        trace: LMTrace,
        *,
        valid: torch.Tensor | None = None,
        update: bool = True,
        learn_hidden: bool = True,
        learn_recurrent: bool = True,
    ) -> dict:
        """跑一整块 ``(B, L)``：内部按位移一位做 next-token。``valid`` 标出非 padding 位置。"""
        total_loss = 0.0
        counted = 0
        fired = 0
        for t in range(tokens.shape[1] - 1):
            target = tokens[:, t + 1]
            if valid is not None:
                target = target.masked_fill(~valid[:, t + 1], -100)
            loss, shots = self.step(
                tokens[:, t],
                target,
                trace,
                update=update,
                learn_hidden=learn_hidden,
                learn_recurrent=learn_recurrent,
            )
            mask = target != -100
            if bool(mask.any()):
                total_loss += float(loss[mask].sum())
                counted += int(mask.sum())
            fired += shots
        return {
            "ce": total_loss / max(counted, 1),
            "tokens": counted,
            "firing_rate": fired / max(counted * self.n_rec, 1),
        }

    @torch.no_grad()
    def evaluate(self, tokens: torch.Tensor, trace: LMTrace, *, valid=None) -> dict:
        """教师强制下的逐步 CE 与 top-1（评测用，不更新任何权重）。"""
        total_loss = 0.0
        hits = 0
        counted = 0
        fired = 0
        for t in range(tokens.shape[1] - 1):
            target = tokens[:, t + 1]
            if valid is not None:
                target = target.masked_fill(~valid[:, t + 1], -100)
            loss, shots = self.step(tokens[:, t], target, trace, update=False)
            mask = target != -100
            if not bool(mask.any()):
                continue
            logits = trace.readout @ self.w_out
            hits += int((logits.argmax(dim=-1)[mask] == target[mask]).sum())
            counted += int(mask.sum())
            total_loss += float(loss[mask].sum())
            fired += shots
        return {
            "ce": total_loss / max(counted, 1),
            "top1": hits / max(counted, 1),
            "tokens": counted,
            "firing_rate": fired / max(counted * self.n_rec, 1),
        }
