"""M3c 群体编码的守卫测试（零依赖）。"""

from __future__ import annotations

import math

import pytest
from biosnn_mcp import ContextCode, ZScoreTopKEncoder


def test_shape_and_binary_spikes():
    encoder = ZScoreTopKEncoder(top_k=4, n_centers=5)
    code = encoder.encode([0.1, -0.3, 2.0, -1.5, 0.0, 0.2])
    assert isinstance(code, ContextCode)
    assert len(code.indices) == 4
    assert len(code.activations) == 4 * 5
    assert len(code.spikes) == 4 * 5
    assert set(code.spikes) <= {0.0, 1.0}


def test_selects_largest_absolute_z():
    encoder = ZScoreTopKEncoder(top_k=2, n_centers=5)
    code = encoder.encode([3.0, -5.0, 0.0])
    assert code.indices == (1, 0)  # |−5| > |3|
    assert code.zscores[0] < 0


def test_constant_embedding_is_all_zero():
    encoder = ZScoreTopKEncoder(top_k=3, n_centers=4)
    code = encoder.encode([1.0, 1.0, 1.0])
    assert code.indices == ()
    assert code.spikes == (0.0,) * 12


def test_deterministic_and_tie_by_index():
    encoder = ZScoreTopKEncoder(top_k=3, n_centers=4)
    first = encoder.encode([1.0, -1.0, 1.0, -1.0])
    second = encoder.encode([1.0, -1.0, 1.0, -1.0])
    assert first == second
    assert first.indices == (0, 1, 2)


def test_centers_cover_span():
    centers = ZScoreTopKEncoder(n_centers=5, span=2.0).centers()
    assert centers[0] == pytest.approx(-2.0)
    assert centers[-1] == pytest.approx(2.0)
    assert len(centers) == 5


def test_threshold_controls_density():
    sparse = ZScoreTopKEncoder(top_k=2, n_centers=9, threshold=0.9).encode([1.0, -1.0, 0.0])
    dense = ZScoreTopKEncoder(top_k=2, n_centers=9, threshold=0.01).encode([1.0, -1.0, 0.0])
    assert sum(sparse.spikes) < sum(dense.spikes)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 0},
        {"n_centers": 1},
        {"sigma": 0.0},
        {"span": -1.0},
        {"threshold": 0.0},
        {"threshold": 1.5},
    ],
)
def test_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        ZScoreTopKEncoder(**kwargs)


def test_rejects_empty_or_nonfinite():
    encoder = ZScoreTopKEncoder()
    for bad in ([], [1.0, math.nan], [math.inf]):
        with pytest.raises(ValueError):
            encoder.encode(bad)
