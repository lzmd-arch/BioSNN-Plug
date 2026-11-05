"""守卫：递推电流的标定必须真的把比值带到目标附近。

S0/S1 的教训：带递推的臂若电流比失控，"学不动"的结论就不能引用。
"""

from __future__ import annotations

import torch

from research.sel.lm.model import TokenEProp

VOCAB, N_REC, FAN_IN = 40, 128, 16


def _tokens() -> torch.Tensor:
    return torch.randint(0, VOCAB, (4, 64), generator=torch.Generator().manual_seed(0))


def test_calibration_hits_target() -> None:
    model = TokenEProp(VOCAB, N_REC, fan_in=FAN_IN, recurrence="random", device="cpu")
    report = model.calibrate_recurrent(_tokens(), target=0.5, tolerance=0.08)
    assert abs(report["achieved"] - 0.5) <= 0.08, report


def test_no_recurrence_reports_zero() -> None:
    model = TokenEProp(VOCAB, N_REC, fan_in=FAN_IN, recurrence="none", device="cpu")
    assert model.measure_current_ratio(_tokens()) == 0.0


def test_ssm_has_no_random_recurrence() -> None:
    model = TokenEProp(VOCAB, N_REC, fan_in=FAN_IN, recurrence="ssm", n_state=8, device="cpu")
    assert model.measure_current_ratio(_tokens()) == 0.0
