"""D2：**两层 LIF，全局部规则**。见 `prereg-lm-d2-depth.md`。

    token → w_in → LIF₁ → w12 → LIF₂ → K 抽头 EMA → w_out → softmax

三处更新全部是局部的，**没有任何 autograd**：
`w_out` ← NLMS；`w12` ← 第 1 层脉冲痕迹 × 第 2 层学习信号；`w_in` ← 输入痕迹 × 第 1 层学习信号。
第 1 层的学习信号由 `post₂ @ w12ᵀ` **对称**回传（真转置，不是随机反馈）。

`depth=1` 不走这个类——就是这个模块之外的原模型，所以"加一层"与"换模型"被构造性地分开了。
真正的守卫是 **`w12=0` 时与 `TokenEProp` 逐位相同**（`tests/test_depth.py`）。
"""

from __future__ import annotations

import math

import torch

__all__ = ["DeepTokenLM"]


class DeepTrace:
    def __init__(self, batch: int, n_rec: int, n_vocab: int, n_taps: int, device, dtype):
        def zero(*shape: int) -> torch.Tensor:
            return torch.zeros(*shape, device=device, dtype=dtype)

        z = zero
        self.v1, self.z1, self.r1 = z(batch, n_rec), z(batch, n_rec), z(batch, n_rec)
        self.v2, self.z2, self.r2 = z(batch, n_rec), z(batch, n_rec), z(batch, n_rec)
        self.eps_in = z(batch, n_vocab)
        self.eps_z1 = z(batch, n_rec)
        self.readouts = z(batch, n_taps, n_rec)


class DeepTokenLM:
    def __init__(
        self,
        n_vocab: int,
        n_rec: int,
        *,
        tau: float = 20.0,
        threshold: float = 0.62,
        dampening_factor: float = 0.3,
        n_refractory: int = 2,
        input_std: float = 0.12,
        lr_in: float = 1e-3,
        lr_mid: float = 1e-3,
        lr_out: float = 0.5,
        w12_scale: float = 1.0,
        readout_taus: tuple[float, ...] = (2.0, 128.0),
        trace_decay: float = 0.0,
        depth: int = 2,
        generator: torch.Generator | None = None,
        device="cpu",
    ) -> None:
        gen = generator or torch.Generator()
        self.device = torch.device(device)
        self.n_vocab, self.n_rec = int(n_vocab), int(n_rec)
        self.threshold = float(threshold)
        self.dampening_factor = float(dampening_factor)
        self.n_refractory = int(n_refractory)
        self.alpha = float(math.exp(-1.0 / tau))
        self.trace_decay = float(trace_decay) if trace_decay > 0 else self.alpha
        self.lr_in, self.lr_mid, self.lr_out = float(lr_in), float(lr_mid), float(lr_out)
        taus = torch.tensor(readout_taus, dtype=torch.float32)
        decays = torch.exp(-1.0 / taus)
        self.readout_decays = decays.to(self.device)
        self.readout_scales = torch.sqrt(1.0 - decays**2).to(self.device)
        #: `depth=1` 时**不经过第 1 层**，`w_in` 直接驱动读出层 —— 这就是原模型，
        #: 也是 D2-0 守卫（与 `TokenEProp` 逐位等价）的被测物。
        self.depth = int(depth)
        self.n_taps = len(readout_taus)
        self.width = self.n_taps * self.n_rec
        self.w_in = (torch.randn(n_vocab, n_rec, generator=gen) * input_std).to(self.device)
        #: `w12_scale` 是本轮**唯一**的可调量：第 2 层电流的初始量级。
        #: 直接照搬 `w_in` 的尺度会让第 2 层电流比输入小几个数量级（第 1 层发放率只有 ~4%）。
        self.w12 = (torch.randn(n_rec, n_rec, generator=gen) / math.sqrt(n_rec) * w12_scale).to(
            self.device
        )
        self.w_out = (torch.randn(self.width, n_vocab, generator=gen) / math.sqrt(self.width)).to(
            self.device
        )

    # -- 内部 ----------------------------------------------------------------

    def _spike(self, new_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scaled = (new_v - self.threshold) / self.threshold
        z = (scaled > 0).to(new_v.dtype)
        psi = (self.dampening_factor / self.threshold) * torch.clamp(1.0 - scaled.abs(), min=0.0)
        return z, psi

    def _advance(self, v, z, r, current):
        new_v = self.alpha * v + current - z * self.threshold
        new_z, psi = self._spike(new_v)
        # F1（2026-11-10）：ψ 的掩码必须用**上一步结束时**的不应期计数器，与 `TokenEProp` 同口径。
        blocked = r > 0
        new_z = torch.where(blocked, torch.zeros_like(new_z), new_z)
        psi = torch.where(blocked, torch.zeros_like(psi), psi)
        new_r = torch.clamp(
            r + self.n_refractory * new_z - 1.0, min=0.0, max=float(self.n_refractory)
        )
        return new_v, new_z, new_r, psi

    # -- 主路径 --------------------------------------------------------------

    def new_trace(self, batch: int) -> DeepTrace:
        return DeepTrace(batch, self.n_rec, self.n_vocab, self.n_taps, self.device, self.w_in.dtype)

    def step(self, tokens, targets, trace: DeepTrace, *, update: bool = True):
        if self.depth == 1:
            # depth=1：**不经过第 1 层**，w_in 直接驱动读出层 —— 这就是原模型
            nv2, z2, r2, psi2 = self._advance(trace.v2, trace.z2, trace.r2, self.w_in[tokens])
            trace.v2, trace.z2, trace.r2 = nv2, z2, r2
            z1, psi1 = z2, psi2
        else:
            nv1, z1, r1, psi1 = self._advance(trace.v1, trace.z1, trace.r1, self.w_in[tokens])
            nv2, z2, r2, psi2 = self._advance(trace.v2, trace.z2, trace.r2, z1 @ self.w12.T)
            trace.v1, trace.z1, trace.r1 = nv1, z1, r1
            trace.v2, trace.z2, trace.r2 = nv2, z2, r2

        trace.eps_in = self.trace_decay * trace.eps_in
        trace.eps_in.scatter_add_(
            1, tokens.reshape(-1, 1), torch.ones_like(tokens, dtype=self.w_in.dtype).reshape(-1, 1)
        )
        trace.eps_z1 = self.trace_decay * trace.eps_z1 + z1
        trace.readouts = trace.readouts * self.readout_decays.view(1, -1, 1) + z2.unsqueeze(1)
        features = (trace.readouts * self.readout_scales.view(1, -1, 1)).reshape(
            tokens.shape[0], -1
        )
        logits = features @ self.w_out
        loss = torch.nn.functional.cross_entropy(
            logits, targets, ignore_index=-100, reduction="none"
        )

        if update:
            ok = (targets != -100).to(features.dtype)
            probability = torch.softmax(logits, dim=-1)
            one_hot = torch.zeros_like(probability)
            one_hot.scatter_(1, targets.clamp(min=0).reshape(-1, 1), 1.0)
            gradient = (probability - one_hot) * ok.unsqueeze(1)
            gradient = gradient / max(tokens.shape[0], 1)

            # **顺序很关键**：学习信号必须用**更新前**的 `w_out` 算（与 `TokenEProp` 一致）。
            # 先更新再回传，等于用新权重解释旧误差 —— 实测让 signal 偏了 ~13%。
            signal2 = (gradient @ self.w_out.T).view(-1, self.n_taps, self.n_rec).sum(dim=1)

            norm_out = features.pow(2).sum(dim=1).mean().clamp(min=1e-3)
            self.w_out.add_(features.T @ gradient / norm_out, alpha=-self.lr_out)
            post2 = signal2 * psi2 * ok.unsqueeze(1)
            # 与 `TokenEProp` 同口径：隐层更新**不再额外除以范数**（范数只进读出的 NLMS）。
            if self.depth == 1:
                # 用与 `TokenEProp` **完全相同**的 einsum 形式：GEMM 与 einsum 的累加顺序不同，
                # 换成 matmul 会在 ~1e-7 量级上分叉，D2-0 就守不住"逐位"。
                self.w_in.add_(torch.einsum("bn,bv->vn", post2, trace.eps_in), alpha=-self.lr_in)
            else:
                # **必须按 eps_z1 的逐列范数归一化**：不归一化时 w12 单调放大，
                # 第 2 层发放率从 4% 涨到 43%、CE 冲到 13.8（实测）。读出用的是 NLMS，
                # 中间层这条之前漏了同一件事。
                norm_mid = trace.eps_z1.pow(2).sum(dim=0).clamp(min=1e-3)
                self.w12.add_(
                    torch.einsum("bn,bm->nm", trace.eps_z1, post2) / norm_mid.unsqueeze(1),
                    alpha=-self.lr_mid,
                )
                signal1 = post2 @ self.w12.T
                post1 = signal1 * psi1 * ok.unsqueeze(1)
                self.w_in.add_(torch.einsum("bn,bv->vn", post1, trace.eps_in), alpha=-self.lr_in)
        return loss, float(z2.sum())

    @torch.no_grad()
    def run(
        self,
        tokens,
        trace: DeepTrace,
        valid=None,
        *,
        update: bool = True,
        learn_hidden: bool = True,
    ) -> dict:
        # D2 目前**不支持**冻结隐层（S11 那条是单层实验）；显式拒绝而不是静默忽略。
        if not learn_hidden:
            raise ValueError("DeepTokenLM 不支持 learn_hidden=False")
        total = torch.zeros((), device=tokens.device)
        counted = torch.zeros((), device=tokens.device)
        for t in range(tokens.shape[1] - 1):
            target = tokens[:, t + 1]
            if valid is not None:
                target = target.masked_fill(~valid[:, t + 1], -100)
            loss, _ = self.step(tokens[:, t], target, trace, update=update)
            mask = (target != -100).to(loss.dtype)
            total = total + (loss * mask).sum()
            counted = counted + mask.sum()
        return {"ce": float(total / counted.clamp(min=1))}

    @torch.no_grad()
    def evaluate(self, tokens, trace, *, valid=None, chunk: int = 256) -> dict:
        """与 ``TokenEProp.evaluate`` 同口径：top-1 / ce / 发放率。"""
        hits = counted = 0
        total = 0.0
        fired = 0.0
        steps = tokens.shape[1] - 1
        for start in range(0, tokens.shape[0], chunk):
            stop = min(start + chunk, tokens.shape[0])
            piece = tokens[start:stop]
            state = self.new_trace(piece.shape[0])
            for t in range(steps):
                target = piece[:, t + 1]
                if valid is not None:
                    target = target.masked_fill(~valid[start:stop, t + 1], -100)
                loss, shots = self.step(piece[:, t], target, state, update=False)
                fired += shots
                keep = target != -100
                if not bool(keep.any()):
                    continue
                features = (state.readouts * self.readout_scales.view(1, -1, 1)).reshape(
                    piece.shape[0], -1
                )
                predicted = (features @ self.w_out).argmax(dim=-1)
                hits += int((predicted[keep] == target[keep]).sum())
                total += float(loss[keep].sum())
                counted += int(keep.sum())
        return {
            "top1": hits / max(counted, 1),
            "ce": total / max(counted, 1),
            "firing_rate": fired / max(tokens.shape[0] * steps, 1),
            "tokens": counted,
        }
