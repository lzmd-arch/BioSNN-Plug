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


def test_delta_readout_learns_a_linear_map():
    from research.cognitive.working_memory import LinearDeltaReadout

    rng = np.random.default_rng(0)
    states = rng.normal(size=(200, 8))
    targets = states @ rng.normal(size=(8, 3))
    readout = LinearDeltaReadout(8, 3, learning_rate=0.05)
    readout.fit(states, targets, epochs=100, batch=32, rng=rng)
    assert average_r2(readout.predict(states), targets) > 0.99


def test_v2_runner_smoke(tmp_path):
    from research.cognitive.run_working_memory_v2 import run as run_v2

    summary = run_v2([0], train_episodes=50, test_episodes=20, out=tmp_path)
    assert "w2_ready" in summary["gate"]
    assert (tmp_path / "summary.json").exists()


def test_nlms_learns_a_linear_map_scale_free():
    from research.cognitive.working_memory import LinearNLMSReadout

    rng = np.random.default_rng(0)
    states = rng.normal(size=(200, 8)) * 10.0
    targets = (states / 10.0) @ rng.normal(size=(8, 3))
    readout = LinearNLMSReadout(8, 3, step=0.5)
    readout.fit(states, targets, epochs=50, rng=rng)
    assert average_r2(readout.predict(states), targets) > 0.99


def test_rls_learns_a_linear_map():
    from research.cognitive.working_memory import LinearRLSReadout

    rng = np.random.default_rng(0)
    states = rng.normal(size=(200, 8))
    targets = states @ rng.normal(size=(8, 3))
    readout = LinearRLSReadout(8, 3)
    readout.fit(states, targets, epochs=2, rng=rng)
    assert average_r2(readout.predict(states), targets) > 0.99


def test_v3_runner_smoke(tmp_path):
    from research.cognitive.run_working_memory_v3 import run as run_v3

    summary = run_v3([0], train_episodes=50, test_episodes=20, out=tmp_path, epochs=5)
    assert "v3_ready" in summary["gate"]
    assert (tmp_path / "summary.json").exists()
