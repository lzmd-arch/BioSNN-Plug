"""工作记忆 v0 的守卫：状态形状、R²、冒烟。"""

from __future__ import annotations

import numpy as np

from research.cognitive.run_working_memory import run
from research.cognitive.working_memory import (
    N_DIMS,
    TAUS,
    average_r2,
    encode_episode,
    fit_ridge,
    predict,
)


def test_state_has_one_block_per_timescale():
    rng = np.random.default_rng(0)
    state = encode_episode(rng.normal(size=N_DIMS), rng.normal(size=N_DIMS), rng=rng)
    assert state.shape == (len(TAUS) * N_DIMS,)


def test_ridge_recovers_a_linear_map():
    rng = np.random.default_rng(1)
    states = rng.normal(size=(100, 8))
    targets = states @ rng.normal(size=(8, 3))
    coefficients = fit_ridge(states, targets)
    assert average_r2(predict(states, coefficients), targets) > 0.999


def test_runner_smoke(tmp_path):
    summary = run([0], train_episodes=50, test_episodes=20, out=tmp_path)
    assert "w0_ready" in summary["gate"]
    assert (tmp_path / "summary.json").exists()
