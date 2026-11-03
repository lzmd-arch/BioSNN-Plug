"""守卫：分区参数"全部相同 + 无区内偏置"时必须与参考实现逐位一致。

这条测试是 `RegionEProp` 能被当作"只改了组织"的证据：只要权重、连接、参数都一样，
它的前向与更新必须与 `EventDrivenEProp` 一模一样，否则实验比的就是实现差异。
"""

from __future__ import annotations

import torch

from research.brainregions.model import RegionEProp, RegionSpec
from research.eprop import EventDrivenEProp

N_IN, N_REC, N_OUT, FAN_IN = 12, 256, 5, 32


def _pair() -> tuple[EventDrivenEProp, RegionEProp]:
    reference = EventDrivenEProp(
        N_IN, N_REC, N_OUT, fan_in=FAN_IN, generator=torch.Generator().manual_seed(1)
    )
    region = RegionEProp(
        N_IN,
        N_REC,
        N_OUT,
        fan_in=FAN_IN,
        regions=[RegionSpec("all", N_REC, tau=20.0, threshold=0.62)],
        intra_fraction=0.0,
        generator=torch.Generator().manual_seed(1),
    )
    with torch.no_grad():
        region.w_in.copy_(reference.w_in)
        region.w_out.copy_(reference.w_out)
        region.w_rec.copy_(reference.w_rec)
    region.pre_index = reference.pre_index
    return reference, region


def test_forward_is_bit_identical() -> None:
    reference, region = _pair()
    inputs = torch.rand(28, 8, N_IN, generator=torch.Generator().manual_seed(3))
    spikes_ref, scaled_ref = reference.simulate(inputs)
    spikes_reg, scaled_reg = region.simulate(inputs)
    assert torch.equal(spikes_ref, spikes_reg)
    assert torch.equal(scaled_ref, scaled_reg)


def test_update_is_bit_identical() -> None:
    reference, region = _pair()
    inputs = torch.rand(28, 8, N_IN, generator=torch.Generator().manual_seed(4))
    targets = torch.arange(8) % N_OUT
    loss_ref = reference.update(inputs, targets)
    loss_reg = region.update(inputs, targets)
    assert loss_ref == loss_reg
    assert torch.equal(reference.w_in, region.w_in)
    assert torch.equal(reference.w_rec, region.w_rec)
    assert torch.equal(reference.w_out, region.w_out)


def test_intra_fraction_keeps_connections_inside_the_region() -> None:
    regions = [RegionSpec("a", 64, 20.0, 0.62), RegionSpec("b", 64, 10.0, 0.50)]
    model = RegionEProp(
        8,
        128,
        3,
        fan_in=32,
        regions=regions,
        intra_fraction=0.75,
        generator=torch.Generator().manual_seed(2),
    )
    assert torch.all(model.pre_index[:64, :24] < 64)
    assert torch.all(model.pre_index[64:, :24] >= 64)


def test_region_counts_must_sum_to_n_rec() -> None:
    try:
        RegionEProp(8, 128, 3, fan_in=8, regions=[RegionSpec("a", 64, 20.0, 0.62)])
    except ValueError:
        return
    raise AssertionError("count 之和不等于 n_rec 时应当报错。")
