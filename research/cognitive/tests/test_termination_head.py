"""终止头 v1 的守卫：特征图、可分数据上的 AUROC、冒烟。"""

from __future__ import annotations

import numpy as np

from research.cognitive.run_termination_head import run
from research.cognitive.termination_head import TerminationHead, build_features
from research.cognitive.world_model import auroc


def test_features_have_the_expected_shape():
    states = np.random.default_rng(0).normal(size=(10, 4))
    actions = np.random.default_rng(1).integers(0, 2, size=10)
    features = build_features(states, actions, states.mean(0), states.std(0) + 1e-6)
    assert features.shape == (10, 4 + 2 + 8 + 4 + 8)


def test_separable_termination_is_ranked_well():
    rng = np.random.default_rng(0)
    states = rng.normal(size=(2000, 4))
    actions = rng.integers(0, 2, size=2000)
    terminated = (states[:, 2] > 1.5).astype(int)
    head = TerminationHead()
    head.fit(states, actions, terminated)
    scores = head.predict_proba(states, actions)
    assert auroc(scores, terminated.astype(int)) > 0.95


def test_runner_smoke(tmp_path):
    summary = run([0], episodes=20, out=tmp_path)
    assert summary["gate"]["auroc_mean"] is not None
    assert (tmp_path / "summary.json").exists()
