"""元认知门控 v0 的守卫：温度、ECE、选择性准确率与 AURC。"""

from __future__ import annotations

import numpy as np

from research.cognitive.metacognitive_gate import (
    ece,
    fit_temperature,
    risk_coverage_auc,
    selective_accuracy,
    softmax,
)


def test_softmax_rows_sum_to_one():
    probabilities = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_temperature_is_positive_and_reduces_nll():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, 5)) * 3.0
    labels = rng.integers(0, 5, size=200)
    temperature = fit_temperature(logits, labels)
    assert temperature > 0.0
    from research.cognitive.metacognitive_gate import _nll

    assert _nll(logits, labels, temperature) <= _nll(logits, labels, 1.0) + 1e-9


def test_ece_is_zero_for_perfect_calibration():
    probabilities = np.array([[0.5, 0.5], [0.5, 0.5]] * 50)
    correct = np.array([True, False] * 50)
    assert ece(probabilities, correct, bins=5) < 1e-12


def test_selective_accuracy_chooses_the_top_scores():
    confidence = np.array([0.9, 0.8, 0.2, 0.1])
    correct = np.array([True, True, False, False])
    accuracy, coverage, _ = selective_accuracy(confidence, correct, coverage=0.5)
    assert accuracy == 1.0
    assert coverage == 0.5


def test_risk_coverage_auc_prefers_better_ranking():
    good_confidence = np.array([0.9, 0.8, 0.2, 0.1])
    bad_confidence = np.array([0.1, 0.2, 0.8, 0.9])
    correct = np.array([True, True, False, False])
    assert risk_coverage_auc(good_confidence, correct) > risk_coverage_auc(bad_confidence, correct)
