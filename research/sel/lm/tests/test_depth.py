"""D2 守卫：两层实现必须**在 `depth=1` 时逐位等于原模型**。

`depth=1` 走的是"`w_in` 直接驱动读出层"，也就是原模型；这条不成立，
"D2 − D1" 量的就不是"加一层"，而是"换了实现"。
"""

from __future__ import annotations

import torch

from research.sel.lm.model import TokenEProp
from research.sel.lm.model_deep import DeepTokenLM

VOCAB, N_REC, STEPS = 37, 16, 24
TAUS = (2.0, 128.0)


def _pair(depth: int):
    # 学习率必须**显式**对齐：两者的默认值不同（TokenEProp 的 lr_in 是 2e-4，
    # DeepTokenLM 是 1e-3）。不对齐时守卫量到的是超参差（正好 5 倍），不是实现差。
    shared = {"input_std": 1.5, "readout_taus": TAUS, "lr_in": 2e-4, "lr_out": 0.5}
    base = TokenEProp(
        VOCAB,
        N_REC,
        fan_in=4,
        recurrence="none",
        pad_vocab=False,
        generator=torch.Generator().manual_seed(0),
        device="cpu",
        **shared,
    )
    deep = DeepTokenLM(
        VOCAB,
        N_REC,
        depth=depth,
        lr_mid=2e-4,
        generator=torch.Generator().manual_seed(1),
        device="cpu",
        **shared,
    )
    deep.w_in.copy_(base.w_in)
    deep.w_out.copy_(base.w_out)
    return base, deep


def test_depth_one_matches_token_eprop() -> None:
    base, deep = _pair(1)
    tokens = torch.randint(0, VOCAB, (3, STEPS), generator=torch.Generator().manual_seed(7))
    expected = base.run(tokens, base.new_trace(3), valid=None, update=False)["ce"]
    got = deep.run(tokens, deep.new_trace(3), update=False)["ce"]
    assert abs(got - expected) < 1e-5, f"depth=1 前向 {got} 与 TokenEProp {expected} 不等"


def test_depth_one_updates_match_token_eprop() -> None:
    """不只是前向：**前 4 步**的权重更新也要一致。

    只比前 4 步，理由与 `test_lowrank.py` 完全相同：阈值是硬判据，
    ~1e-7 的浮点差在若干步后会翻转一个脉冲，之后两条轨迹**按定义**就不该逐位相同。
    实测第 1 步 w_out 差 4e-9、第 24 步累积到 5e-6 —— 那是混沌放大，不是实现差。
    """
    base, deep = _pair(1)
    tokens = torch.randint(0, VOCAB, (3, 5), generator=torch.Generator().manual_seed(8))
    base.run(tokens, base.new_trace(3), valid=None)
    deep.run(tokens, deep.new_trace(3))
    assert torch.allclose(base.w_out, deep.w_out, atol=1e-6), "w_out 更新不一致"
    # `atol` 必须 ≥ fp32 在 |w|≈1.5 处的 ulp（1.8e-7）；早先写 1e-7 是**不可达的判据**，
    # 卡住的是最后一位数字（-8.4626e-01 对 -8.4627e-01），不是实现差。
    assert torch.allclose(base.w_in, deep.w_in, atol=1e-6), "w_in 更新不一致"


def test_depth_two_reaches_layer_two() -> None:
    """depth=2 时第 2 层必须真的在发放，且 w12 真的在动——否则 D2 测的是空壳。"""
    _, deep = _pair(2)
    tokens = torch.randint(0, VOCAB, (3, STEPS), generator=torch.Generator().manual_seed(9))
    before = deep.w12.clone()
    trace = deep.new_trace(3)
    deep.run(tokens, trace)
    assert float(trace.z2.sum()) + float(trace.z1.sum()) > 0, "两层都没有发放"
    assert (deep.w12 - before).abs().max() > 0, "w12 没更新"
