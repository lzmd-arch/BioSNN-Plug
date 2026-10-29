"""D10 守卫：判定表按修正 1（A 用随机方向标定）走，原阈值只作并列记录。"""

from __future__ import annotations

from research.rstdp_spiking.diagnose_direction import _verdict


def _inputs(*, spread: float, random_median: float, c16: float, b_range: float):
    drive = {"spread": spread}
    amplification = {"16": {"range": c16}}
    transfer = {"dynamic_range": b_range}
    return drive, amplification, transfer, {"median": random_median}


def test_direction_needs_twice_the_random_spread() -> None:
    """ "比随机方向更随状态变"是 A 的全部内容；2× 是阈值，1.99× 不算过。"""
    drive, amp, transfer, control = _inputs(spread=0.006, random_median=0.003, c16=1.0, b_range=2.0)
    assert _verdict(drive, amp, transfer, control)["direction"] is True

    drive, amp, transfer, control = _inputs(
        spread=0.0059, random_median=0.003, c16=1.0, b_range=2.0
    )
    verdict = _verdict(drive, amp, transfer, control)
    assert verdict["direction"] is False
    assert verdict["reason"] == "other_unattributed"  # C 过、A 不过 → 不硬归因


def test_raw_threshold_is_reported_but_does_not_decide() -> None:
    """A_spread 0.06 过了原阈值，但只有随机的 1.2 倍 → 仍判 A 不成立。"""
    drive, amp, transfer, control = _inputs(spread=0.06, random_median=0.05, c16=0.1, b_range=2.0)
    verdict = _verdict(drive, amp, transfer, control)
    assert verdict["a_spread_raw_pass"] is True
    assert verdict["direction"] is False
    assert verdict["reason"] == "no_differential_direction"


def test_zero_random_spread_does_not_explode() -> None:
    """随机方向极差为 0 时不能除零；给一个极小的分母并让判定仍然确定。"""
    drive, amp, transfer, control = _inputs(spread=0.0, random_median=0.0, c16=0.0, b_range=0.0)
    verdict = _verdict(drive, amp, transfer, control)
    assert verdict["direction"] is False
    assert verdict["reason"] == "no_differential_direction"


def test_all_three_branches_of_the_table() -> None:
    drive, amp, transfer, control = _inputs(spread=0.01, random_median=0.003, c16=0.0, b_range=0.2)
    verdict = _verdict(drive, amp, transfer, control)
    assert verdict["reason"] == "lif_saturated"
    assert verdict["saturated"] is True

    drive, amp, transfer, control = _inputs(spread=0.01, random_median=0.003, c16=1.0, b_range=2.0)
    verdict = _verdict(drive, amp, transfer, control)
    assert verdict["reason"] == "amplitude_too_small"
