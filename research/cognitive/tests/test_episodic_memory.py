"""情景记忆 v0 的守卫：稀疏键、外积、字面基线。"""

from __future__ import annotations

import numpy as np

from research.cognitive.episodic_memory import (
    HebbianEpisodicMemory,
    RawNearestNeighbor,
    SparseEncoder,
    build_values,
)
from research.cognitive.run_episodic_memory import run


def test_encoder_is_sparse_and_deterministic():
    states = np.random.default_rng(0).normal(size=(5, 4))
    first = SparseEncoder(dim=64, k=8, seed=0).encode(states)
    second = SparseEncoder(dim=64, k=8, seed=0).encode(states)

    assert first.shape == (5, 64)
    assert np.array_equal(first, second)
    assert np.all(np.sum(first != 0, axis=1) == 8)
    assert set(np.unique(first)) <= {-1.0, 0.0, 1.0}


def test_values_encode_action_termination_and_time():
    values = build_values(np.array([0, 1]), np.array([False, True]), np.array([0.0, 10.0]))

    assert values.shape == (2, 5)
    assert values[0, 0] == 1.0 and values[1, 1] == 1.0
    assert values[0, 2] == 1.0 and values[1, 3] == 1.0
    assert np.allclose(values[:, 4], [0.0, 1.0])


def test_hebbian_memory_recalls_one_event():
    memory = HebbianEpisodicMemory()
    keys = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    values = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    memory.store(keys, values)

    assert np.array_equal(memory.recall(keys)[0], [1.0, 0.0, 0.0])


def test_raw_nearest_neighbor_finds_the_clean_state():
    nearest = RawNearestNeighbor()
    states = np.array([[0.0, 0.0], [10.0, 0.0]])
    values = np.array([[1.0, 0.0], [0.0, 1.0]])
    nearest.store(states, values)

    assert np.array_equal(nearest.recall(np.array([[0.1, 0.0]]))[0], [1.0, 0.0])


def test_runner_smoke_writes_summary(tmp_path):
    summary = run([0], episodes=10, out=tmp_path, capacities=(20, 40))

    assert summary["gate"]["v0_ready"] is False
    assert len(summary["per_seed"]) == 2
    assert (tmp_path / "summary.json").exists()


def test_v1_memory_separates_time_and_recalls_one_event():
    from research.cognitive.episodic_memory import EpisodicMemoryV1

    memory = EpisodicMemoryV1(sparsity=2)
    keys = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    values = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    times = np.array([0.2, 0.8])
    memory.store(keys, values, times)

    recalled_values, recalled_times = memory.recall(keys, attractor_steps=0)
    assert np.array_equal(recalled_values[0], values[0])
    assert np.allclose(recalled_times, times)


def test_pseudo_inverse_recalls_stored_values():
    from research.cognitive.episodic_memory import PseudoInverseMemory

    memory = PseudoInverseMemory()
    keys = np.array([[1.0, 0.0], [0.0, 1.0]])
    values = np.array([[1.0, 0.0], [0.0, 1.0]])
    memory.store(keys, values)
    assert np.allclose(memory.recall(keys), values)


def test_v1_runner_smoke(tmp_path):
    from research.cognitive.run_episodic_memory_v1 import run as run_v1

    summary = run_v1([0], episodes=10, out=tmp_path, capacities=(20,))
    assert summary["gate"]["v1_ready"] is False
    assert len(summary["per_seed"]) == 4


def test_dense_encoders_have_expected_forms():
    from research.cognitive.episodic_memory import SparseEncoder

    states = np.random.default_rng(0).normal(size=(5, 4))
    dense_sign = SparseEncoder(dim=16, k=16, seed=0, mode="dense_sign").encode(states)
    dense_cont = SparseEncoder(dim=16, k=16, seed=0, mode="dense").encode(states)

    assert dense_sign.shape == (5, 16)
    assert set(np.unique(dense_sign)) <= {-1.0, 1.0}
    assert dense_cont.shape == (5, 16)
    assert np.all(np.isfinite(dense_cont))


def test_v2_runner_smoke(tmp_path):
    from research.cognitive.run_episodic_memory_v2 import run as run_v2

    summary = run_v2([0], episodes=10, out=tmp_path, capacities=(20,))
    assert summary["gate"]["v2_ready"] is False
    assert len(summary["per_seed"]) == 4 + 2
