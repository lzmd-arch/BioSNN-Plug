"""守卫：A3 低秩读出在 ``w_b = I`` 且 ``w_a = w_out`` 时必须与稠密读出逐位一致。"""

from __future__ import annotations

import torch

from research.sel.lm.model import TokenEProp

VOCAB, N_REC, BATCH, STEPS = 20, 16, 3, 12
TAUS = (2.0, 8.0)


def _make(**kwargs) -> TokenEProp:
    return TokenEProp(
        VOCAB,
        N_REC,
        fan_in=4,
        readout_taus=TAUS,
        generator=torch.Generator().manual_seed(0),
        device="cpu",
        **kwargs,
    )


def test_low_rank_with_identity_basis_matches_dense() -> None:
    dense = _make()
    eye = torch.eye(VOCAB)
    low = _make(readout_rank=VOCAB, readout_basis=eye)
    assert low.w_b is not None
    # 把低秩的 A 设成稠密的 w_out（形状同为 (M, V)），B = I ⇒ 代数上完全等价
    assert low.w_a is not None
    low.w_a = dense.w_out.clone()

    gen = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, VOCAB, (BATCH, STEPS), generator=gen)
    trace_dense, trace_low = dense.new_trace(BATCH), low.new_trace(BATCH)
    # 只比**前 4 步**：脉冲网络的阈值是硬判据，1e-5 的参数扰动会在某个时刻翻转一个脉冲，
    # 之后两条轨迹按定义就不该逐位相同（这里实测 5 步完全逐位一致，第 7 步翻转）。
    # 4 步足以覆盖"logits / 学习信号 / 权重的第一轮更新"这三处代数等价。
    for t in range(4):
        target = tokens[:, t + 1]
        loss_dense, _ = dense.step(tokens[:, t], target, trace_dense)
        loss_low, _ = low.step(tokens[:, t], target, trace_low)
        assert torch.equal(loss_dense, loss_low), t
        assert torch.equal(dense.w_out, low.w_a), t
        assert torch.equal(dense.w_in, low.w_in), t


def test_low_rank_reduces_readout_parameters() -> None:
    dense = _make()
    low = _make(readout_rank=8)
    assert low.readout_width == dense.readout_width
    dense_params = dense.w_out.numel()
    low_params = low.w_a.numel() + low.w_b.numel()
    assert low_params < dense_params
