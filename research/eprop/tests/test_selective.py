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


def test_loss_mode_with_full_fraction_matches_baseline():
    """选 100% 时间步时，loss 模式必须与 baseline 更新逐位相同。"""
    inputs = (torch.rand(8, 2, 4, generator=torch.Generator().manual_seed(2)) < 0.5).float()
    targets = torch.tensor([0, 2])
    baseline = _learner()
    selective = _learner(selective_fraction=1.0, selective_mode="loss")
    baseline.update(inputs, targets)
    selective.update(inputs, targets)
    assert torch.equal(baseline.cell.w_in, selective.cell.w_in)
    assert torch.equal(baseline.cell.w_rec, selective.cell.w_rec)
    assert torch.equal(baseline.w_out, selective.w_out)


def test_loss_mode_tracks_requested_share():
    learner = _learner(selective_fraction=0.3, selective_mode="loss")
    inputs = (torch.rand(20, 3, 4, generator=torch.Generator().manual_seed(3)) < 0.5).float()
    learner.update(inputs, torch.tensor([0, 1, 2]))
    assert 0.15 <= learner.last_update_fraction <= 0.45


def test_loss_mode_selects_different_steps_than_input_mode():
    """两种准则选出的时间步不应完全相同，否则「按损失选」没有落到实处。"""
    inputs = (torch.rand(12, 2, 4, generator=torch.Generator().manual_seed(4)) < 0.5).float()
    targets = torch.tensor([0, 1])
    by_input = _learner(selective_fraction=0.3, selective_mode="input")
    by_loss = _learner(selective_fraction=0.3, selective_mode="loss")
    by_input.update(inputs, targets)
    by_loss.update(inputs, targets)
    assert not torch.equal(by_input.cell.w_rec, by_loss.cell.w_rec)


def test_rejects_unknown_mode():
    try:
        EPropLearner(2, 2, 2, selective_mode="energy")
    except ValueError:
        return
    raise AssertionError("未知 selective_mode 应被拒绝")
