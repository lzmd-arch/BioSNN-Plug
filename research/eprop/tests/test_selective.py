"""事件优先级（选择性时间步）的最小守卫。"""

from __future__ import annotations

import torch

from research.eprop.eprop import EPropLearner


def _learner(**kwargs):
    return EPropLearner(4, 6, 3, beta=0.0, generator=torch.Generator().manual_seed(0), **kwargs)


def test_default_updates_all_steps():
    learner = _learner()
    inputs = torch.zeros(5, 2, 4)
    learner.update(inputs, torch.tensor([0, 1]))
    assert learner.last_update_fraction == 1.0


def test_selective_fraction_tracks_requested_share():
    learner = _learner(selective_fraction=0.4)
    generator = torch.Generator().manual_seed(1)
    inputs = (torch.rand(10, 3, 4, generator=generator) < 0.5).float()
    learner.update(inputs, torch.tensor([0, 1, 2]))
    assert 0.2 <= learner.last_update_fraction <= 0.61


def test_rejects_invalid_fraction():
    for value in (0.0, -0.1, 1.1):
        try:
            EPropLearner(2, 2, 2, selective_fraction=value)
        except ValueError:
            continue
        raise AssertionError(f"selective_fraction={value} 应被拒绝")
