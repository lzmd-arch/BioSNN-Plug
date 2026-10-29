"""M3d 睡眠巩固协议的守卫测试。"""

from __future__ import annotations

import pytest

from biosnn_mcp import ConsolidationReport, MemoryItem, SleepConsolidator


def _item(identifier: str, vector: tuple[float, ...]) -> MemoryItem:
    return MemoryItem(id=identifier, text=identifier, embedding=vector)


def test_identical_entries_merge():
    items = [_item("a", (1.0, 0.0, 0.0)), _item("b", (1.0, 0.0, 0.0))]
    survivors, report = SleepConsolidator(noise_sigma=0.0, merge_threshold=0.99).consolidate(items)
    assert len(survivors) == 1
    assert survivors[0].id == "a"
    assert ("a", "b") in report.merged_pairs
    assert isinstance(report, ConsolidationReport)


def test_distant_entries_stay_separate():
    items = [_item("a", (1.0, 0.0, 0.0)), _item("b", (0.0, 1.0, 0.0))]
    survivors, report = SleepConsolidator(noise_sigma=0.0, merge_threshold=0.9).consolidate(items)
    assert len(survivors) == 2
    assert report.merged_pairs == []


def test_deterministic_with_seed():
    items = [_item("a", (1.0, 0.1, 0.0)), _item("b", (1.0, 0.12, 0.0))]
    first, report_a = SleepConsolidator(noise_sigma=0.05, seed=7).consolidate(items)
    second, report_b = SleepConsolidator(noise_sigma=0.05, seed=7).consolidate(items)
    assert first == second
    assert report_a.merged_pairs == report_b.merged_pairs


def test_noise_scale_is_relative_to_signal():
    """σ 是**总**扰动比例：小 σ 时回放余弦应接近 1（曾因逐维 σ·‖x‖ 被压到 0.58）。"""
    dimension = 128
    vector = tuple(1.0 / (dimension**0.5) for _ in range(dimension))
    item = _item("a", vector)
    _, report = SleepConsolidator(noise_sigma=0.05, merge_threshold=-1.0).consolidate([item])
    assert report.mean_cosine_drift > 0.995


def test_empty_input():
    survivors, report = SleepConsolidator().consolidate([])
    assert survivors == []
    assert report.original_count == 0


def test_duplicate_ids_rejected():
    items = [_item("a", (1.0, 0.0)), _item("a", (0.0, 1.0))]
    with pytest.raises(ValueError):
        SleepConsolidator().consolidate(items)


@pytest.mark.parametrize(
    "kwargs",
    [{"noise_sigma": -0.1}, {"noise_sigma": 1.5}, {"merge_threshold": 1.5}, {"rounds": 0}],
)
def test_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        SleepConsolidator(**kwargs)


def test_bad_embedding_rejected():
    with pytest.raises(ValueError):
        MemoryItem(id="a", text="a", embedding=())
