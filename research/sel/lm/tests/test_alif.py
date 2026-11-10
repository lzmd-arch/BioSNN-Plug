"""守卫：G1 ALIF——`beta=0` 与 LIF 逐位一致；`beta>0` 时适应变量真的在动。"""

from __future__ import annotations

import torch

from research.sel.lm.model import TokenEProp

VOCAB, N_REC, BATCH, STEPS = 20, 16, 3, 10


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


def test_beta_zero_matches_lif() -> None:
    base, same = _make(), _make(beta=0.0)
    gen = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, VOCAB, (BATCH, STEPS), generator=gen)
    tb, ts = base.new_trace(BATCH), same.new_trace(BATCH)
    for t in range(STEPS - 1):
        lb, _ = base.step(tokens[:, t], tokens[:, t + 1], tb)
        ls, _ = same.step(tokens[:, t], tokens[:, t + 1], ts)
        assert torch.equal(lb, ls)
    assert torch.equal(base.w_in, same.w_in)
    assert ts.a is None


def test_adaptation_variable_accumulates() -> None:
    model = _make(beta=0.5)
    gen = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, VOCAB, (BATCH, STEPS), generator=gen)
    trace = model.new_trace(BATCH)
    assert trace.a is not None
    for t in range(STEPS - 1):
        model.step(tokens[:, t], tokens[:, t + 1], trace)
    assert float(trace.a.abs().sum()) > 0.0
