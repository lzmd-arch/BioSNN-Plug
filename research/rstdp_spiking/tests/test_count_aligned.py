"""count-aligned 资格的符号守卫。"""

from __future__ import annotations

import torch

from research.rstdp_spiking.plasticity import STDPWindow, build_trace


def test_selected_column_positive_unselected_negative():
    trace = build_trace("count-aligned", 3, 2, decay=0.9)
    pre = torch.zeros(2, 1, 3)
    pre[0, 0, :] = 1.0
    post = torch.ones(2, 1, 2)
    policy = torch.tensor([0.5, 0.5])
    action = 0
    trace.accumulate_window(pre, post, STDPWindow(), action, policy=policy)
    assert torch.all(trace.value[:, action] > 0)
    assert torch.all(trace.value[:, 1 - action] < 0)


def test_negative_signal_moves_opposite_way():
    trace = build_trace("count-aligned", 3, 2, decay=0.9)
    pre = torch.zeros(2, 1, 3)
    pre[0, 0, :] = 1.0
    post = torch.ones(2, 1, 2)
    policy = torch.tensor([0.5, 0.5])
    trace.accumulate_window(pre, post, STDPWindow(), 0, policy=policy)
    weights = torch.zeros(3, 2)
    trace.apply(weights, 1.0, -1.0)
    assert torch.all(weights[:, 0] < 0)
    assert torch.all(weights[:, 1] > 0)


def test_vacuous_window_is_zero():
    trace = build_trace("count-aligned", 3, 2, decay=0.9)
    pre = torch.zeros(2, 1, 3)
    post = torch.zeros(2, 1, 2)
    trace.accumulate_window(pre, post, STDPWindow(), 0, policy=torch.tensor([0.5, 0.5]))
    assert torch.count_nonzero(trace.value) == 0
