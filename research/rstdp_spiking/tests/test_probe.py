"""探针记录（D5/D7/D8 共用）的守卫测试。"""

from __future__ import annotations

import torch

from research.components.cartpole_encoding import CARTPOLE_HALF_WIDTHS, population_centers
from research.rstdp_spiking.actor import SpikingActor
from research.rstdp_spiking.cartpole import _probe_actor


def _actor_and_centers(seed: int = 0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    centers = population_centers(
        8, half_widths=CARTPOLE_HALF_WIDTHS, generator=generator, device=torch.device("cpu")
    )
    actor = SpikingActor(
        8, 2, steps=5, trace_rule="count-aligned", generator=generator, device=torch.device("cpu")
    )
    return actor, centers


def test_default_probe_keys_and_shapes():
    actor, centers = _actor_and_centers()
    record = _probe_actor(actor, centers, 1.0, seed=0, episode=100, repeats=3)
    assert record["episode"] == 100
    for label in ("pos", "neg"):
        assert len(record[f"counts_{label}"]) == 2
        assert record[f"delta_sd_{label}"] >= 0.0
        assert 0.0 <= record[f"p_left_{label}"] <= 1.0
    assert 0.0 <= record["zero_window_fraction"] <= 1.0
    assert len(record["column_l1"]) == 2


def test_custom_angle_grid():
    actor, centers = _actor_and_centers()
    angles = tuple((round(-0.2 + 0.1 * index, 2), f"a{index}") for index in range(5))
    record = _probe_actor(actor, centers, 1.0, seed=0, episode=100, repeats=2, angles=angles)
    for index in range(5):
        assert f"counts_a{index}" in record
        assert f"delta_mean_a{index}" in record
    assert "counts_pos" not in record
