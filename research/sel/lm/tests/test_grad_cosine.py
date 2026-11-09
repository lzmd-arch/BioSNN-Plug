"""守卫：读出滤波的反向修正（F2）的前提与效果。

- **无滤波时必须是 no-op**：把 ``tau_read`` 调到 ≈0，反向 EMA 修正不得改变方向
  （否则说明它补的不是滤波那一项）；
- **有滤波时必须有正向效果**：``tau_read=8`` 时 cos 至少改善 0.01。

这两条把「F2 补的是读出滤波的反向路径」这件事钉死，防止将来改滤波器口径时静默破坏它。
"""

from __future__ import annotations

import math

import torch

from research.sel.lm.grad_cosine import DiffLM, bptt_delta, cosine, eprop_delta

VOCAB, N_REC, STEPS, BATCH = 30, 64, 24, 4
COMMON = {
    "tau": 20.0,
    "threshold": 0.62,
    "dampening": 0.3,
    "n_refractory": 2.0,
}


def _inputs():
    gen = torch.Generator().manual_seed(0)
    w_in = torch.randn(VOCAB, N_REC, generator=gen) * 0.12
    w_out = torch.randn(N_REC, VOCAB, generator=gen) / math.sqrt(N_REC)
    tokens = torch.randint(0, VOCAB, (BATCH, STEPS), generator=gen)
    targets = tokens.clone()
    targets[:, 0] = -100
    return w_in, w_out, tokens, targets


def _cosines(tau_read: float) -> tuple[float, float]:
    w_in, w_out, tokens, targets = _inputs()
    plain = eprop_delta(tokens, targets, w_in, w_out, tau_read=tau_read, fixed_mask=True, **COMMON)
    fixed = eprop_delta(
        tokens,
        targets,
        w_in,
        w_out,
        tau_read=tau_read,
        fixed_mask=True,
        backward_ema=True,
        **COMMON,
    )
    model = DiffLM(w_in, w_out, tau_read=tau_read, fixed_mask=True, **COMMON)
    grad = bptt_delta(model, tokens, targets, STEPS)
    return cosine(plain, grad), cosine(fixed, grad)


def test_bwd_ema_is_noop_without_readout_filter() -> None:
    """读出无滤波时，反向 EMA 修正必须退化为 no-op。"""
    plain, corrected = _cosines(0.01)
    assert abs(corrected - plain) < 1e-9, (plain, corrected)


def test_bwd_ema_improves_cosine_with_readout_filter() -> None:
    """读出有滤波（τ=8）时，反向 EMA 必须让方向更接近真梯度。"""
    plain, corrected = _cosines(8.0)
    assert corrected > plain + 0.01, (plain, corrected)
