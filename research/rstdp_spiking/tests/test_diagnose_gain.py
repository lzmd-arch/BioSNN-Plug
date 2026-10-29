"""D11 守卫：构造的差方向必须真的量到目标相对幅度（否则整轮标定是无标尺的）。"""

from __future__ import annotations

import torch

from research.components.cartpole_encoding import encode_state, population_centers
from research.rstdp_spiking.diagnose_gain import (
    HALF_WIDTHS,
    N_FEATURES,
    SIGMA,
    _common_column,
    _grid_states,
    _scale_delta,
)


def _actual_relative(column: torch.Tensor, delta: torch.Tensor, centers: torch.Tensor) -> float:
    weight = torch.stack([column + delta / 2, column - delta / 2], dim=1)
    differences = []
    drives = []
    for state in _grid_states():
        features = encode_state(state, centers, SIGMA)
        projected = features @ weight
        differences.append(float(projected[0] - projected[1]))
        drives.append((abs(float(projected[0])) + abs(float(projected[1]))) / 2)
    return (max(differences) - min(differences)) / (sum(drives) / len(drives))


def test_scaled_direction_hits_the_target_relative_amplitude() -> None:
    seed = 0
    centers = population_centers(
        N_FEATURES, half_widths=HALF_WIDTHS, generator=torch.Generator().manual_seed(seed)
    )
    column = _common_column(N_FEATURES, seed)
    direction = torch.empty(N_FEATURES).normal_(
        generator=torch.Generator().manual_seed(seed * 7919 + 3)
    )
    direction = direction / direction.norm()

    for rho in (0.05, 0.10):
        delta = _scale_delta(column, direction, rho, centers)
        assert abs(_actual_relative(column, delta, centers) - rho) < 1e-3


def test_zero_rho_gives_exactly_identical_columns() -> None:
    centers = population_centers(
        N_FEATURES, half_widths=HALF_WIDTHS, generator=torch.Generator().manual_seed(0)
    )
    column = _common_column(N_FEATURES, 0)
    direction = torch.ones(N_FEATURES)
    delta = _scale_delta(column, direction, 0.0, centers)
    assert torch.equal(delta, torch.zeros_like(delta))


def test_common_column_is_l1_normalised() -> None:
    """与 Actor 的 `_normalize` 同口径；否则「相对幅度」的定义会跟着漂。"""
    column = _common_column(N_FEATURES, 0)
    assert abs(float(column.abs().sum()) - 1.0) < 1e-6
