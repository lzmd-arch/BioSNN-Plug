"""守卫：多时间尺度读出（R4）的形状、互斥与 EMA 语义。

R3 那把守卫抓到过一次真错（`signal` 仍是 `(B, K·N)` 而 `psi` 是 `(B, N)`），
这里把同一类错误写成断言：形状对但信号全 0 的静默失败必须变红。
"""

from __future__ import annotations

import pytest
import torch

from research.sel.lm.model import TokenEProp

VOCAB, N_REC, FAN_IN = 40, 128, 16
TAUS = (2.0, 8.0, 32.0, 128.0)


def _tokens(steps: int = 16) -> torch.Tensor:
    return torch.randint(0, VOCAB, (4, steps), generator=torch.Generator().manual_seed(0))


def test_multiscale_widens_the_readout() -> None:
    model = TokenEProp(VOCAB, N_REC, fan_in=FAN_IN, readout_taus=TAUS, device="cpu")
    assert model.w_out.shape == (len(TAUS) * N_REC, VOCAB)


def test_multiscale_excludes_readout_width() -> None:
    with pytest.raises(ValueError):
        TokenEProp(
            VOCAB,
            N_REC,
            fan_in=FAN_IN,
            readout_width=32,
            readout_taus=TAUS,
            device="cpu",
        )


def test_multiscale_first_step_readout_is_z() -> None:
    """EMA 从零起：第一步的每个抽头都等于该步的脉冲。"""
    model = TokenEProp(VOCAB, N_REC, fan_in=FAN_IN, readout_taus=TAUS, device="cpu")
    tokens = _tokens(1)
    trace = model.new_trace(tokens.shape[0])
    model.run(tokens, trace)
    assert trace.readouts is not None
    assert trace.readouts.shape == (tokens.shape[0], len(TAUS), N_REC)
    for tap in range(len(TAUS)):
        assert torch.equal(trace.readouts[:, tap], trace.z)


def test_multiscale_learns() -> None:
    """学习信号按抽头求和之后，输入权重必须真的动。"""
    model = TokenEProp(VOCAB, N_REC, fan_in=FAN_IN, readout_taus=TAUS, device="cpu")
    before = model.w_in.clone()
    tokens = _tokens()
    for _ in range(5):
        model.run(tokens, model.new_trace(tokens.shape[0]))
    assert not torch.equal(model.w_in, before)
