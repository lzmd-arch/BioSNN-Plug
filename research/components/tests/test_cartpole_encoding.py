"""CartPole 编码的逐位守卫。

两条线的已发表数字都建立在各自的中心公式上，而现在的实现是共用的。这里把
**两条线的历史写法**当 oracle：共用实现与它们必须 ``torch.equal``，不是近似。
这也是这个模块存在的理由——共享实现只有在逐位等价时才允许替换复制代码。
"""

from __future__ import annotations

import torch

from research.components.cartpole_encoding import (
    CARTPOLE_HALF_WIDTHS,
    encode_state,
    population_centers,
)


def _draw(seed: int, n: int = 500) -> torch.Tensor:
    return torch.rand(n, 4, generator=torch.Generator().manual_seed(seed))


def test_rate_line_centers_are_bit_identical_to_its_frozen_formula():
    """速率线：``((u − 0.5) × 3.0)``（`research/rstdp/cartpole.py` 的历史写法）。"""
    actual = population_centers(
        500, half_widths=(1.5, 1.5, 1.5, 1.5), generator=torch.Generator().manual_seed(0)
    )
    expected = (_draw(0) - 0.5) * 3.0
    assert torch.equal(actual, expected)


def test_spiking_line_centers_are_bit_identical_to_its_frozen_formula():
    """脉冲线：``((u × 2 − 1) × STATE_SCALES)``（`rstdp_spiking/encoding.py` 的历史写法，已并入本模块）。"""
    scales = torch.tensor(CARTPOLE_HALF_WIDTHS)
    actual = population_centers(
        500, half_widths=CARTPOLE_HALF_WIDTHS, generator=torch.Generator().manual_seed(0)
    )
    expected = (_draw(0) * 2.0 - 1.0) * scales
    assert torch.equal(actual, expected)


def test_centers_stay_within_the_requested_half_widths():
    centers = population_centers(
        1000, half_widths=CARTPOLE_HALF_WIDTHS, generator=torch.Generator().manual_seed(1)
    )
    limit = torch.tensor(CARTPOLE_HALF_WIDTHS)
    assert bool((centers.abs() <= limit).all())


def test_centers_land_on_the_requested_device():
    centers = population_centers(
        8, half_widths=(1.5,) * 4, generator=torch.Generator().manual_seed(0), device=None
    )
    assert centers.device.type == "cpu"


def test_encode_state_matches_the_gaussian_formula():
    centers = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    state = torch.tensor([0.0, 0.0])
    sigma = 0.5
    out = encode_state(state, centers, sigma)
    assert torch.allclose(
        out, torch.tensor([1.0, float(torch.exp(torch.tensor(-1.0 / (2 * sigma**2))))])
    )


def test_encode_state_rejects_non_positive_sigma():
    centers = torch.zeros(3, 4)
    for sigma in (0.0, -1.0):
        try:
            encode_state(torch.zeros(4), centers, sigma)
        except ValueError as error:
            assert "sigma" in str(error)
        else:  # pragma: no cover
            raise AssertionError(f"sigma={sigma} 应被拒绝")
