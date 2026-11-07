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


def test_readout_width_full_is_bit_identical() -> None:
    """``readout_width=n_rec`` 必须与默认逐位一致——否则 R3 的对照不是单变量。"""
    base = TokenEProp(VOCAB, N_REC, fan_in=FAN_IN, device="cpu")
    same = TokenEProp(VOCAB, N_REC, fan_in=FAN_IN, readout_width=N_REC, device="cpu")
    assert torch.equal(base.w_out, same.w_out)


def test_readout_width_narrow_shrinks_only_the_readout() -> None:
    """窄读出只改 ``w_out``，神经元数与输入权重都不动。"""
    base = TokenEProp(VOCAB, N_REC, fan_in=FAN_IN, device="cpu")
    narrow = TokenEProp(VOCAB, N_REC, fan_in=FAN_IN, readout_width=32, device="cpu")
    assert narrow.w_out.shape == (32, VOCAB)
    assert narrow.w_in.shape == base.w_in.shape
    assert torch.equal(narrow.w_in, base.w_in)
    tokens = _tokens()
    narrow.run(tokens, narrow.new_trace(4))  # 形状不对会抛
