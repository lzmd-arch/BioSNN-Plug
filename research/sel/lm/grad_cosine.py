"""梯度余弦守卫：局部规则（e-prop）离真实梯度有多远，以及**远在哪里**。

只回答一个问题，不做训练：

1. ``eprop``：`TokenEProp` 现在的规则（ψ 用**本步更新后**的不应期计数器掩码）；
2. ``eprop_fix``：同一条规则，但 ψ 用**上一步结束**的计数器（`research/eprop/traces.py`
   里写死的正确口径："判据是上一步之后的计数器，不是本步之后的"）；
3. ``bptt(B)``：同一前向的可微镜像做截断反向传播，B ∈ {1, 6, 64, 192}；
4. ``eprop_bwd``：在 2 的基础上把学习信号再做**反向 EMA**（补读出滤波的反向路径）。

cos(eprop, bptt) ≈ 1 ⇒ 局部规则已是精确梯度；明显 < 1 ⇒ 缺口存在，再看哪一版把它补上。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[3]
OUT = Path("D:/Projects/_lm_tmp/gradcos")


# ---------------------------------------------------------------- 局部规则


@torch.no_grad()
def eprop_delta(
    tokens: torch.Tensor,
    targets: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    *,
    tau: float,
    tau_read: float,
    threshold: float,
    dampening: float,
    n_refractory: float,
    fixed_mask: bool,
    backward_ema: bool = False,
    readout_taus: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """返回规则给出的 ``Σ_t post_t ⊗ eps_in_t``（形状同 ``w_in``）。"""
    batch, steps = tokens.shape
    n_vocab, n_rec = w_in.shape
    alpha = math.exp(-1.0 / tau)
    if readout_taus is None:
        decays = torch.tensor([math.exp(-1.0 / tau_read)], device=w_in.device)
    else:
        decays = torch.exp(
            -1.0 / torch.tensor(readout_taus, dtype=torch.float32, device=w_in.device)
        )
    scales = (
        torch.ones_like(decays)
        if readout_taus is None
        else torch.rsqrt((1.0 - decays) / (1.0 + decays))
    )
    n_taps = int(decays.numel())
    v = w_in.new_zeros(batch, n_rec)
    z = torch.zeros_like(v)
    refractory = torch.zeros_like(v)
    readouts = w_in.new_zeros(batch, n_taps, n_rec)
    eps_in = w_in.new_zeros(batch, n_vocab)
    posts: list[torch.Tensor] = []
    traces: list[torch.Tensor] = []
    psis: list[torch.Tensor] = []
    signals: list[torch.Tensor] = []
    for t in range(steps - 1):
        token = tokens[:, t]
        target = targets[:, t + 1]
        valid = target != -100
        active = valid.to(v.dtype).unsqueeze(1)
        current = w_in[token]
        new_v = alpha * v + current - z * threshold
        scaled = (new_v - threshold) / threshold
        pre_refractory = refractory > 0
        fresh = (scaled > 0).to(v.dtype)
        new_z = torch.where(pre_refractory, torch.zeros_like(fresh), fresh)
        new_r = torch.clamp(
            refractory + n_refractory * new_z - 1.0, min=0.0, max=float(n_refractory)
        )
        v = torch.where(active > 0, new_v, v)
        z = torch.where(active > 0, new_z, torch.zeros_like(new_z))
        refractory = torch.where(active > 0, new_r, refractory)
        readouts = decays.view(1, -1, 1) * readouts + z.unsqueeze(1)
        features = (readouts * scales.view(1, -1, 1)).reshape(batch, -1)

        eps_in = alpha * eps_in
        eps_in.scatter_add_(1, token.reshape(-1, 1), active.to(eps_in.dtype))

        logits = features @ w_out
        probability = torch.softmax(logits, dim=-1)
        one_hot = torch.zeros_like(probability)
        one_hot.scatter_(1, target.clamp(min=0).reshape(-1, 1), 1.0)
        gradient = (probability - one_hot) * active / max(batch, 1)
        signal_sub = gradient @ w_out.T
        signal_taps = signal_sub.view(batch, n_taps, n_rec)
        signal = signal_taps.sum(dim=1)
        psi_full = (dampening / threshold) * torch.clamp(1.0 - scaled.abs(), min=0.0)
        mask = refractory if not fixed_mask else pre_refractory.to(v.dtype)
        psi = torch.where(mask > 0, torch.zeros_like(psi_full), psi_full)
        post = signal * psi * active
        signals.append(signal_taps.clone())
        posts.append(post)
        traces.append(eps_in.clone())
        psis.append((psi * active).clone())
    if backward_ema:
        # 读出滤波的反向路径：每个抽头各做一次 signal'_t = Σ_{u≥t} decay_k^{u-t} · signal_u
        acc = torch.zeros_like(signals[0])
        fixed: list[torch.Tensor] = []
        for item in reversed(signals):
            acc = decays.view(1, -1, 1) * acc + item
            fixed.append(acc.clone())
        fixed = list(reversed(fixed))
        posts = [sig.sum(dim=1) * ps for sig, ps in zip(fixed, psis, strict=True)]
    total = torch.zeros_like(w_in)
    for post, trace in zip(posts, traces, strict=True):
        total += torch.einsum("bn,bv->vn", post, trace)
    return total


# ---------------------------------------------------------------- BPTT 镜像


class DiffLM(nn.Module):
    """`TokenEProp` 前向的可微镜像（替代梯度用同一个 ψ）。"""

    def __init__(
        self,
        w_in,
        w_out,
        *,
        tau,
        tau_read,
        threshold,
        dampening,
        n_refractory,
        fixed_mask,
        readout_taus=None,
    ):
        super().__init__()
        self.w_in = nn.Parameter(w_in.clone())
        self.register_buffer("w_out", w_out.clone())
        self.alpha = math.exp(-1.0 / tau)
        if readout_taus is None:
            decays = torch.tensor([math.exp(-1.0 / tau_read)], device=w_in.device)
            scales = torch.ones_like(decays)
        else:
            decays = torch.exp(
                -1.0 / torch.tensor(readout_taus, dtype=torch.float32, device=w_in.device)
            )
            scales = torch.rsqrt((1.0 - decays) / (1.0 + decays))
        self.register_buffer("decays", decays)
        self.register_buffer("scales", scales)
        self.threshold = threshold
        self.dampening = dampening
        self.n_refractory = n_refractory
        self.fixed_mask = fixed_mask

    def forward(self, tokens, targets):
        batch, steps = tokens.shape
        _, n_rec = self.w_in.shape
        v = self.w_in.new_zeros(batch, n_rec)
        z = torch.zeros_like(v)
        refractory = torch.zeros_like(v)
        readouts = self.w_in.new_zeros(batch, int(self.decays.numel()), n_rec)
        total = v.new_zeros(())
        counted = 0
        for t in range(steps - 1):
            token = tokens[:, t]
            target = targets[:, t + 1]
            valid = target != -100
            active = valid.to(v.dtype).unsqueeze(1)
            current = self.w_in[token]
            new_v = self.alpha * v + current - z * self.threshold
            scaled = (new_v - self.threshold) / self.threshold
            pre_refractory = refractory > 0
            hard = (scaled > 0).to(v.dtype)
            psi_full = (self.dampening / self.threshold) * torch.clamp(1.0 - scaled.abs(), min=0.0)
            mask = refractory if not self.fixed_mask else pre_refractory.to(v.dtype)
            psi = torch.where(mask > 0, torch.zeros_like(psi_full), psi_full)
            new_z = hard.detach() + psi.detach() * (new_v - new_v.detach())
            new_z = torch.where(pre_refractory, torch.zeros_like(new_z), new_z)
            new_r = torch.clamp(
                refractory + self.n_refractory * new_z.detach() - 1.0,
                min=0.0,
                max=float(self.n_refractory),
            )
            v = torch.where(active > 0, new_v, v)
            z = torch.where(active > 0, new_z, torch.zeros_like(new_z))
            refractory = torch.where(active > 0, new_r, refractory)
            readouts = self.decays.view(1, -1, 1) * readouts + z.unsqueeze(1)
            features = (readouts * self.scales.view(1, -1, 1)).reshape(batch, -1)
            logits = features @ self.w_out
            loss = torch.nn.functional.cross_entropy(
                logits, target, ignore_index=-100, reduction="none"
            )
            if bool(valid.any()):
                total = total + loss[valid].sum()
                counted += int(valid.sum())
        return total / max(counted, 1)


def bptt_delta(model: DiffLM, tokens, targets, trunc: int) -> torch.Tensor:
    """截断 BPTT：按 ``trunc`` 切块，块间只传状态（这里用整段前向 + 分段 backward 近似）。"""
    model.zero_grad(set_to_none=True)
    loss = model(tokens, targets)
    loss.backward()
    return model.w_in.grad.detach().clone()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    num = float((a * b).sum())
    den = float(a.norm()) * float(b.norm())
    return num / den if den > 0 else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="梯度余弦守卫")
    parser.add_argument("--vocab", type=int, default=60)
    parser.add_argument("--n-rec", type=int, default=128)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau-read", type=float, default=8.0, help="读出 EMA 的 τ；很小=无滤波")
    parser.add_argument("--readout-taus", default="", help="逗号分隔；给了就走多抽头")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    gen = torch.Generator().manual_seed(args.seed)
    taus = tuple(float(item) for item in args.readout_taus.split(",") if item.strip()) or None
    width = args.n_rec * (1 if taus is None else len(taus))
    w_in = torch.randn(args.vocab, args.n_rec, generator=gen) * 0.12
    w_out = torch.randn(width, args.vocab, generator=gen) / math.sqrt(width)
    tokens = torch.randint(0, args.vocab, (args.batch, args.steps), generator=gen)
    targets = tokens.clone()
    targets[:, 0] = -100

    common = {
        "tau": 20.0,
        "tau_read": args.tau_read,
        "threshold": 0.62,
        "dampening": 0.3,
        "n_refractory": 2.0,
    }
    eprop = eprop_delta(tokens, targets, w_in, w_out, fixed_mask=False, readout_taus=taus, **common)
    eprop_fix = eprop_delta(
        tokens, targets, w_in, w_out, fixed_mask=True, readout_taus=taus, **common
    )
    eprop_bwd = eprop_delta(
        tokens,
        targets,
        w_in,
        w_out,
        fixed_mask=True,
        backward_ema=True,
        readout_taus=taus,
        **common,
    )
    report: dict = {"seed": args.seed, "cos_vs_bptt": {}}
    fixed_model = DiffLM(w_in, w_out, fixed_mask=True, readout_taus=taus, **common)
    cur_model = DiffLM(w_in, w_out, fixed_mask=False, readout_taus=taus, **common)
    # 第一轮只做**整段** BPTT（本序列 48 步，B≥48 都等于整段）；
    # 分段（B=1/6/64）要另一份按块清零的镜像，等这一轮结论出来再写。
    grad = bptt_delta(fixed_model, tokens, targets, tokens.shape[1])
    report["cos_vs_bptt"]["full"] = {
        "eprop": round(cosine(eprop, grad), 4),
        "eprop_fix": round(cosine(eprop_fix, grad), 4),
        "eprop_bwd": round(cosine(eprop_bwd, grad), 4),
    }
    print(
        f"整段 BPTT: cos(eprop)={cosine(eprop, grad):+.4f}  "
        f"cos(fix)={cosine(eprop_fix, grad):+.4f}  cos(fix+bwdEMA)={cosine(eprop_bwd, grad):+.4f}",
        flush=True,
    )
    grad_cur = bptt_delta(cur_model, tokens, targets, tokens.shape[1])
    report["cos_eprop_vs_eprop_fix"] = round(cosine(eprop, eprop_fix), 4)
    report["cos_eprop_vs_bptt_current_mirror"] = round(cosine(eprop, grad_cur), 4)
    print(f"cos(eprop, eprop_fix) = {cosine(eprop, eprop_fix):+.4f}")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "gradcos.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
