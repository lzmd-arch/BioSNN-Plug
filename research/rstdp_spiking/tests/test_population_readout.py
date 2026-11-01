"""群体线性解码读出：拟合正确性、正则方向、以及"它确实能解出线性映射"。"""

from __future__ import annotations

import torch

from research.rstdp_spiking.population_readout import RidgeReadout


def test_recovers_a_known_linear_map():
    """给一个真实的线性映射，解码器应当把它解出来（这是它该有的能力）。"""
    torch.manual_seed(0)
    n_neurons, n_actions, n_samples = 32, 3, 400
    true_w = torch.randn(n_neurons, n_actions)
    counts = torch.rand(n_samples, n_neurons)
    targets = counts @ true_w
    decoder = RidgeReadout(alpha=1e-5).fit(counts, targets)
    assert torch.allclose(decoder.scores(counts), targets, atol=1e-2)


def test_alpha_shrinks_the_solution():
    torch.manual_seed(1)
    counts = torch.rand(50, 16)
    targets = torch.rand(50, 2)
    weak = RidgeReadout(alpha=0.0).fit(counts, targets)
    strong = RidgeReadout(alpha=10.0).fit(counts, targets)
    assert strong.weight.norm() < weak.weight.norm()


def test_rejects_bad_shapes_and_alpha():
    import pytest

    decoder = RidgeReadout()
    with pytest.raises(ValueError):
        decoder.fit(torch.rand(4, 8), torch.rand(3, 2))
    with pytest.raises(ValueError):
        RidgeReadout(alpha=-1.0)
    with pytest.raises(RuntimeError):
        RidgeReadout().scores(torch.rand(1, 4))


def test_choose_returns_argmax():
    torch.manual_seed(2)
    counts = torch.rand(20, 8)
    targets = torch.zeros(20, 3)
    targets[:, 1] = 1.0  # 永远选 1
    decoder = RidgeReadout(alpha=1e-3).fit(counts, targets)
    assert decoder.choose(counts[0]) == 1
