"""``summarise_synergy`` 的算术测试：判据怎么算，就在测试里怎么钉住。"""

from __future__ import annotations

import pytest

from research.arbiter.summarise_synergy import SYNERGY_DELTA, quartiles, summarise


def _row(arm: str, seed: int, fisher: float, *, gate: bool = True) -> dict:
    return {"arm": arm, "seed": seed, "fisher": fisher, "gate_passed": gate}


def _groups(mix: dict[int, float], h: dict[int, float], e: dict[int, float]) -> dict:
    return {
        "H": [_row("H", s, v) for s, v in h.items()],
        "E": [_row("E", s, v) for s, v in e.items()],
        "MIX": [_row("MIX", s, v) for s, v in mix.items()],
    }


def test_quartiles_simple_series() -> None:
    q1, median, q3 = quartiles([1.0, 2.0, 3.0, 4.0, 5.0])
    assert median == pytest.approx(3.0)
    assert q1 == pytest.approx(2.0)
    assert q3 == pytest.approx(4.0)


def test_quartiles_empty_is_nan() -> None:
    q1, median, q3 = quartiles([])
    assert q1 != q1 and median != median and q3 != q3  # 全为 NaN


def test_gate_failed_seeds_are_dropped_and_reported() -> None:
    groups = _groups(
        mix={0: 0.60, 1: 0.61, 2: 0.62},
        h={0: 0.47, 1: 0.47, 2: 0.47},
        e={0: 0.32, 1: 0.32, 2: 0.32},
    )
    groups["H"][1]["gate_passed"] = False  # seed 1 的 H 作废

    result = summarise(groups)

    assert result["per_group"]["H"]["seeds_used"] == [0, 2]
    assert result["per_group"]["H"]["seeds_dropped"] == [1]
    # 三种子交集里少了 1，协同量只剩 0 和 2
    assert result["synergy"]["seeds_used"] == [0, 2]


def test_synergy_is_paired_against_best_arm() -> None:
    groups = _groups(
        mix={0: 0.70, 1: 0.72, 2: 0.68, 3: 0.69, 4: 0.71},
        h={0: 0.50, 1: 0.48, 2: 0.52, 3: 0.49, 4: 0.51},
        e={0: 0.60, 1: 0.62, 2: 0.58, 3: 0.61, 4: 0.63},
    )

    result = summarise(groups)

    # 协同量必须扣掉**较强**的那一臂（seed 0 这里是 E 的 0.60），不是 H 的 0.50
    assert result["synergy"]["per_seed"][0] == pytest.approx(0.70 - 0.60)
    assert result["criterion"]["delta_ok"] is True


def test_criterion_fails_when_synergy_below_delta() -> None:
    groups = _groups(
        mix=dict.fromkeys(range(5), 0.52),
        h=dict.fromkeys(range(5), 0.5),
        e=dict.fromkeys(range(5), 0.5),
    )

    result = summarise(groups)

    assert result["synergy"]["median"] == pytest.approx(0.02)
    assert result["criterion"]["delta_ok"] is False
    assert result["criterion"]["passed"] is False


def test_separation_requires_mix_lower_quartile_above_worst_arm_upper_quartile() -> None:
    # MIX 中位数够高，但波动很大，下四分位压不住 H 的上四分位
    groups = _groups(
        mix={0: 0.90, 1: 0.40, 2: 0.95, 3: 0.42, 4: 0.88},
        h=dict.fromkeys(range(5), 0.45),
        e=dict.fromkeys(range(5), 0.3),
    )

    result = summarise(groups)

    assert result["synergy"]["median"] >= SYNERGY_DELTA
    assert result["criterion"]["delta_ok"] is True
    assert result["criterion"]["separation_ok"] is False
    assert result["criterion"]["passed"] is False


def test_missing_group_is_reported_not_crashed() -> None:
    result = summarise({"MIX": [_row("MIX", 0, 0.5)]})
    assert "缺少" in result["criterion"]["status"]


def test_controls_are_reported_relative_to_mix() -> None:
    groups = _groups(
        mix=dict.fromkeys(range(5), 0.7),
        h=dict.fromkeys(range(5), 0.5),
        e=dict.fromkeys(range(5), 0.5),
    )
    groups["MIX_frozen"] = [_row("MIX", s, 0.55) for s in range(5)]
    groups["MIX_shuffle"] = [_row("MIX", s, 0.20) for s in range(5)]

    result = summarise(groups)

    assert result["controls"]["MIX_frozen"]["vs_mix"] == pytest.approx(-0.15)
    assert result["controls"]["MIX_shuffle"]["vs_mix"] == pytest.approx(-0.50)
