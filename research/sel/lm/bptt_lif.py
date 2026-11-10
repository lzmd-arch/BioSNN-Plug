"""R12 探针：**同一个单层 LIF，用 BPTT 训练**（真梯度）。

它补的是这张 2×2 的缺格：

|            | 局部规则            | BPTT        |
| :--------- | :------------------ | :---------- |
| 单层 LIF   | 0.3090（2 抽头 16k）| **???**     |
| GRU        | —                   | 0.5318      |

- `local-SNN` 与 `BPTT-SNN` 之差 = **规则的账**（架构相同）
- `BPTT-SNN` 与 `BPTT-GRU` 之差 = **架构的账**（规则相同）

**前向必须与 `TokenEProp` 逐位等价**——否则这个对照测的就不是同一个模型。
等价守卫在 `tests/test_bptt_equiv.py`。反传用的伪导数与 `TokenEProp` 的 ψ 同口径：
``dz/dv = dampening/threshold · clamp(1 − |scaled|, 0)``。
"""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["BPTTLIF"]


class BPTTLIF(nn.Module):
    """与 ``TokenEProp``（``recurrence="none"`` + 多抽头读出）同架构的 LIF，但全程可反传。"""

    def __init__(
        self,
        n_vocab: int,
        n_rec: int,
        *,
        tau: float = 20.0,
        input_std: float = 0.12,
        threshold: float = 0.62,
        dampening_factor: float = 0.3,
        n_refractory: int = 2,
        readout_taus: tuple[float, ...] = (2.0, 8.0, 32.0, 128.0),
        generator: torch.Generator | None = None,
        device="cpu",
    ) -> None:
        super().__init__()
        gen = generator or torch.Generator()
        self.n_vocab = int(n_vocab)
        self.n_rec = int(n_rec)
        self.threshold = float(threshold)
        self.dampening_factor = float(dampening_factor)
        self.n_refractory = int(n_refractory)
        self.alpha = float(math.exp(-1.0 / tau))
        taus = torch.tensor(readout_taus, dtype=torch.float32)
        decays = torch.exp(-1.0 / taus)
        #: 与 `TokenEProp` 逐位同式：稳态方差归一化用 sqrt(1 − d²)
        self.register_buffer("readout_decays", decays)
        self.register_buffer("readout_scales", torch.sqrt(1.0 - decays**2))
        self.n_taps = len(readout_taus)
        self.width = self.n_taps * self.n_rec
        self.w_in = nn.Parameter(torch.randn(n_vocab, n_rec, generator=gen) * input_std)
        self.w_out = nn.Parameter(
            torch.randn(self.width, n_vocab, generator=gen) / math.sqrt(self.width)
        )
        self.to(torch.device(device))

    # -- 前向 ----------------------------------------------------------------

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor | None = None):
        """返回 ``(逐样本 CE, target)``；图保留到 ``w_in`` 与 ``w_out``。"""
        batch, length = tokens.shape
        dtype = self.w_in.dtype
        v = torch.zeros(batch, self.n_rec, device=tokens.device, dtype=dtype)
        z = torch.zeros_like(v)
        refractory = torch.zeros_like(v)
        readouts = torch.zeros(batch, self.n_taps, self.n_rec, device=tokens.device, dtype=dtype)
        losses = []
        logit_list = []
        for t in range(length - 1):
            target = tokens[:, t + 1]
            if valid is not None:
                target = target.masked_fill(~valid[:, t + 1], -100)
            active = (target != -100).to(dtype).unsqueeze(1)
            current = self.w_in[tokens[:, t]]
            new_v = self.alpha * v + current - z * self.threshold
            scaled = (new_v - self.threshold) / self.threshold
            hard = (scaled > 0).to(dtype)
            # 伪导数与 TokenEProp 的 ψ 同口径：dz/dv = dampening/θ · clamp(1−|scaled|, 0)
            surrogate = self.dampening_factor * torch.clamp(1.0 - scaled.abs(), min=0.0)
            fresh = hard.detach() + surrogate * (scaled - scaled.detach())
            new_z = fresh * (1.0 - (refractory > 0).to(dtype))
            new_r = torch.clamp(
                refractory + self.n_refractory * new_z - 1.0,
                min=0.0,
                max=float(self.n_refractory),
            )
            if valid is not None:
                new_v = torch.where(active > 0, new_v, v)
                new_z = torch.where(active > 0, new_z, torch.zeros_like(new_z))
                new_r = torch.where(active > 0, new_r, refractory)
            v, z, refractory = new_v, new_z, new_r
            readouts = readouts * self.readout_decays.view(1, -1, 1) + z.unsqueeze(1)
            features = (readouts * self.readout_scales.view(1, -1, 1)).reshape(batch, -1)
            logits = features @ self.w_out
            logit_list.append(logits)
            losses.append(
                torch.nn.functional.cross_entropy(
                    logits, target, ignore_index=-100, reduction="none"
                )
            )
        return torch.stack(losses, dim=1), torch.stack(logit_list, dim=1)

    @torch.no_grad()
    def evaluate(
        self, tokens: torch.Tensor, valid: torch.Tensor | None = None, chunk: int = 64
    ) -> dict:
        """分块评测：避免把 (B×L×V) 的 logits 一次全materialize。"""
        total_ce = 0.0
        hits = 0
        counted = 0
        for start in range(0, tokens.shape[0], chunk):
            stop = min(start + chunk, tokens.shape[0])
            tok = tokens[start:stop]
            val = None if valid is None else valid[start:stop]
            losses, logits = self.forward(tok, val)
            targets = torch.stack([tok[:, t + 1] for t in range(tok.shape[1] - 1)], dim=1)
            if val is not None:
                masks = torch.stack([val[:, t + 1] for t in range(tok.shape[1] - 1)], dim=1)
                targets = targets.masked_fill(~masks, -100)
            keep = targets != -100
            total_ce += float(losses[keep].sum())
            predicted = logits.argmax(dim=-1)
            hits += int((predicted[keep] == targets[keep]).sum())
            counted += int(keep.sum())
        return {"ce": total_ce / max(counted, 1), "top1": hits / max(counted, 1), "tokens": counted}
