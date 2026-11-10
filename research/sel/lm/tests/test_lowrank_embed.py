"""守卫：A2 低秩输入在 ``proj = I`` 且 ``emb = w_in`` 时必须与稠密输入逐位一致。"""

from __future__ import annotations

import torch

from research.sel.lm.model import TokenEProp

VOCAB, N_REC, BATCH, STEPS = 20, 16, 3, 12


def _make(**kwargs) -> TokenEProp:
    return TokenEProp(
        VOCAB,
        N_REC,
        fan_in=4,
        readout_taus=(2.0, 8.0),
        generator=torch.Generator().manual_seed(0),
        device="cpu",
        **kwargs,
    )


def test_factorised_input_with_identity_projection_matches_dense() -> None:
    dense = _make()
    eye = torch.eye(N_REC)
    low = _make(embed_rank=N_REC, embed_proj=eye)
    assert low.emb is not None
    low.emb = dense.w_in.clone()  # (V, N) —— 与 proj=I 合起来正是 w_in

    gen = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, VOCAB, (BATCH, STEPS), generator=gen)
    trace_dense, trace_low = dense.new_trace(BATCH), low.new_trace(BATCH)
    # 脉冲网络的硬阈值会让 1e-5 的扰动翻转脉冲，所以只比前 4 步（与 A3 的守卫同一口径）。
    for t in range(4):
        target = tokens[:, t + 1]
        loss_dense, _ = dense.step(tokens[:, t], target, trace_dense)
        loss_low, _ = low.step(tokens[:, t], target, trace_low)
        assert torch.equal(loss_dense, loss_low), t
        assert torch.equal(dense.w_in, low.emb), t
        assert torch.equal(dense.w_out, low.w_out), t


def test_frozen_projection_is_not_updated() -> None:
    low = _make(
        embed_rank=8, embed_proj=torch.randn(8, N_REC, generator=torch.Generator().manual_seed(1))
    )
    assert low.proj is not None and low.emb is not None
    before = low.proj.clone()
    gen = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, VOCAB, (BATCH, STEPS), generator=gen)
    trace = low.new_trace(BATCH)
    for t in range(STEPS - 1):
        low.step(tokens[:, t], tokens[:, t + 1], trace)
    assert torch.equal(low.proj, before)
    assert not torch.equal(low.emb, torch.zeros_like(low.emb))
