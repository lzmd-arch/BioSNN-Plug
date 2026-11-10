"""守卫：G2 异质膜 τ——`tau_groups=1` 与默认逐位一致；分组后 α 与 τ 对得上。"""

from __future__ import annotations

import math

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


def test_single_group_matches_default() -> None:
    base, same = _make(), _make(tau_groups=1)
    gen = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, VOCAB, (BATCH, STEPS), generator=gen)
    tb, ts = base.new_trace(BATCH), same.new_trace(BATCH)
    for t in range(STEPS - 1):
        lb, _ = base.step(tokens[:, t], tokens[:, t + 1], tb)
        ls, _ = same.step(tokens[:, t], tokens[:, t + 1], ts)
        assert torch.equal(lb, ls)
    assert torch.equal(base.w_in, same.w_in)


def test_groups_share_alpha_and_trace_shape() -> None:
    model = _make(tau_groups=4)
    assert model.tau_vec is not None and model.alpha_vec is not None
    assert model.tau_vec.numel() == N_REC
    assert torch.allclose(model.alpha_vec, torch.exp(-1.0 / model.tau_vec))
    assert len(model.group_ids) == 4
    trace = model.new_trace(BATCH)
    assert trace.eps_in_groups is not None
    assert trace.eps_in_groups.shape == (4, BATCH, VOCAB)
    # 每组一个 τ（冻结），且极值与 log 网格对得上
    taus = sorted({round(float(x), 6) for x in model.tau_vec})
    assert len(taus) == 4
    assert math.isclose(taus[0], 5.0, rel_tol=1e-4) and math.isclose(taus[-1], 500.0, rel_tol=1e-4)


def test_groups_run_and_traces_accumulate() -> None:
    """分组路径真的跑起来了——判据用**痕迹**而不是权重（随机初始化下可能一个脉冲都不发）。"""
    model = _make(tau_groups=4)
    gen = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, VOCAB, (BATCH, STEPS), generator=gen)
    trace = model.new_trace(BATCH)
    for t in range(STEPS - 1):
        model.step(tokens[:, t], tokens[:, t + 1], trace)
    assert trace.eps_in_groups is not None
    assert float(trace.eps_in_groups.abs().sum()) > 0.0
