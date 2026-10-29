"""D10 守卫：判定表按修正 1/2 走——A 用无饱和的差驱动极差对随机方向标定。"""

from __future__ import annotations

import torch

from research.rstdp_spiking.diagnose_direction import _relative_drive, _verdict


def _inputs(*, spread_abs: float, random_median_abs: float, c16: float, b_range: float):
    drive = {"difference_spread_abs": spread_abs, "spread": 0.0}
    amplification = {"16": {"range": c16}}
    transfer = {"dynamic_range": b_range}
    return drive, amplification, transfer, {"median_abs": random_median_abs, "median": 0.0}


def test_direction_needs_twice_the_random_spread() -> None:
    """ "比随机方向更随状态变"是 A 的全部内容；2× 是阈值，1.99× 不算过。"""
    drive, amp, transfer, control = _inputs(
        spread_abs=0.006, random_median_abs=0.003, c16=1.0, b_range=2.0
    )
    assert _verdict(drive, amp, transfer, control)["direction"] is True

    drive, amp, transfer, control = _inputs(
        spread_abs=0.0059, random_median_abs=0.003, c16=1.0, b_range=2.0
    )
    verdict = _verdict(drive, amp, transfer, control)
    assert verdict["direction"] is False
    assert verdict["reason"] == "other_unattributed"  # C 过、A 不过 → 不硬归因


def test_ratio_saturation_does_not_decide_anything() -> None:
    """修正 2 的核心：比值版本 `spread` 恒为 0 时，判定只看无饱和的差驱动极差。"""
    drive, amp, transfer, control = _inputs(
        spread_abs=0.01, random_median_abs=0.002, c16=0.1, b_range=2.0
    )
    drive["spread"] = 0.0  # 比值被钉死
    verdict = _verdict(drive, amp, transfer, control)
    assert verdict["direction"] is True
    assert verdict["a_norm"] == 5.0


def test_zero_random_spread_does_not_explode() -> None:
    """随机方向极差为 0 时不能除零；给一个极小的分母并让判定仍然确定。"""
    drive, amp, transfer, control = _inputs(
        spread_abs=0.0, random_median_abs=0.0, c16=0.0, b_range=0.0
    )
    verdict = _verdict(drive, amp, transfer, control)
    assert verdict["direction"] is False
    assert verdict["reason"] == "no_differential_direction"


def test_all_three_branches_of_the_table() -> None:
    drive, amp, transfer, control = _inputs(
        spread_abs=0.01, random_median_abs=0.003, c16=0.0, b_range=0.2
    )
    verdict = _verdict(drive, amp, transfer, control)
    assert verdict["reason"] == "lif_saturated"
    assert verdict["saturated"] is True

    drive, amp, transfer, control = _inputs(
        spread_abs=0.01, random_median_abs=0.003, c16=1.0, b_range=2.0
    )
    verdict = _verdict(drive, amp, transfer, control)
    assert verdict["reason"] == "amplitude_too_small"


def test_relative_drive_saturates_exactly_at_opposite_drive_signs() -> None:
    """把修正 2 的动机钉成断言：两列驱动异号 ⇒ r 恒为 ±2，但差驱动极差仍有信息。"""
    features = torch.ones(4)
    same_sign = torch.tensor([[0.5, 0.25], [0.5, 0.25], [0.0, 0.0], [0.0, 0.0]])
    opposite = torch.tensor([[0.5, -0.5], [0.5, -0.5], [0.0, 0.0], [0.0, 0.0]])

    assert abs(_relative_drive(features, same_sign)["r"]) < 2.0
    assert _relative_drive(features, opposite)["r"] == 2.0

    # 异号时把一列整体放大，比值仍是 2（饱和），而差驱动跟着变（有信息）。
    scaled = opposite.clone()
    scaled[:, 1] *= 2.0
    assert _relative_drive(features, scaled)["r"] == 2.0
    assert (
        _relative_drive(features, scaled)["difference"]
        != _relative_drive(features, opposite)["difference"]
    )
