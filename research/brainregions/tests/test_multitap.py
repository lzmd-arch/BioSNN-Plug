"""守卫：``taps=1`` 的多抽头读出必须与参考实现逐位一致。

A3 的结论要能说成"只改了读出"，这条测试是它的证据：抽头数为 1 时，
前向、读出、更新后的每一个权重都不能与 ``EventDrivenEProp`` 差一位。
"""

from __future__ import annotations

import torch

from research.brainregions.model import RegionSpec
from research.brainregions.multitap import MultiTapEProp, tap_indices
from research.eprop import EventDrivenEProp

N_IN, N_REC, N_OUT, FAN_IN = 12, 256, 5, 32


def _pair(taps: int = 1) -> tuple[EventDrivenEProp, MultiTapEProp]:
    reference = EventDrivenEProp(
        N_IN, N_REC, N_OUT, fan_in=FAN_IN, generator=torch.Generator().manual_seed(1)
    )
    model = MultiTapEProp(
        N_IN,
        N_REC,
        N_OUT,
        taps=taps,
        fan_in=FAN_IN,
        regions=[RegionSpec("all", N_REC, tau=20.0, threshold=0.62)],
        intra_fraction=0.0,
        generator=torch.Generator().manual_seed(1),
    )
    with torch.no_grad():
        model.w_in.copy_(reference.w_in)
        if taps == 1:
            model.w_out.copy_(reference.w_out)
        model.w_rec.copy_(reference.w_rec)
    model.pre_index = reference.pre_index
    return reference, model


def test_tap_indices_cover_both_ends() -> None:
    assert tap_indices(28, 1) == [27]
    assert tap_indices(28, 28) == list(range(28))
    assert tap_indices(28, 7) == [0, 4, 9, 13, 18, 22, 27]
    assert tap_indices(5, 3) == [0, 2, 4]


def test_single_tap_forward_and_readout_are_bit_identical() -> None:
    reference, model = _pair(taps=1)
    inputs = torch.rand(28, 8, N_IN, generator=torch.Generator().manual_seed(3))
    spikes_ref, scaled_ref = reference.simulate(inputs)
    spikes_reg, scaled_reg = model.simulate(inputs)
    assert torch.equal(spikes_ref, spikes_reg)
    assert torch.equal(scaled_ref, scaled_reg)
    assert torch.equal(reference.predict(inputs), model.predict(inputs))


def test_single_tap_update_is_bit_identical() -> None:
    reference, model = _pair(taps=1)
    inputs = torch.rand(28, 8, N_IN, generator=torch.Generator().manual_seed(4))
    targets = torch.arange(8) % N_OUT
    loss_ref = reference.update(inputs, targets)
    loss_reg = model.update(inputs, targets)
    assert loss_ref == loss_reg
    assert torch.equal(reference.w_in, model.w_in)
    assert torch.equal(reference.w_rec, model.w_rec)
    assert torch.equal(reference.w_out, model.w_out)


def test_readout_width_scales_with_taps() -> None:
    _, model = _pair(taps=4)
    assert model.w_out.shape == (4 * N_REC, N_OUT)
    inputs = torch.rand(28, 8, N_IN, generator=torch.Generator().manual_seed(5))
    assert model.predict(inputs).shape == (8,)
    model.update(inputs, torch.arange(8) % N_OUT)  # 形状不对会在这里抛
