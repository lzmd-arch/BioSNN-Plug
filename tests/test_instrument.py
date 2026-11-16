"""仪器守卫的负向测试。

**为什么给检查写测试**：一个永远通过的检查比没有检查更糟——它给人一种已经守住了的错觉。
2026-11-16 一天之内，同一类工程错误让约 12 轮实验作废，所以这里**每条都先造出问题**，
断言它真的被抓到。
"""

from __future__ import annotations

# ruff: noqa: N806  —— 线性代数里 X/W/A/C 是惯例
import itertools
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.instrument import (
    FIRING_WINDOW,
    assert_patched,
    calibrate,
    linear_probe_acc,
    scale_grid,
    validity,
)


def test_scale_grid_is_geometric_and_dense_enough():
    """网格必须够密：实测 ring 在 1.0→0.0027、2.0→0.311，粗网格会跨过窗口。"""
    g = scale_grid(0.4, 12.0, 1.15)
    assert g[0] <= 0.5 and g[-1] >= 10
    ratios = [b / a for a, b in itertools.pairwise(g)]
    assert all(abs(r - 1.15) < 1e-3 for r in ratios), "ratio 不是常数"


def test_calibrate_finds_a_valid_scale():
    """造一个只在 scale∈[1.1,1.4] 落进窗口的假网络。"""

    def fake(sc):
        return 0.08 if 1.1 <= sc <= 1.4 else (0.0 if sc < 1.1 else 0.6)

    ok, _sc, fr = calibrate(fake)
    assert ok and FIRING_WINDOW[0] <= fr <= FIRING_WINDOW[1]


def test_calibrate_rejects_silent_network():
    """**静默网络必须被判无效** —— 这是当天最贵的错误。"""
    ok, best, fr = calibrate(lambda _sc: 0.0001)
    assert not ok and best is None and math.isnan(fr)


def test_calibrate_rejects_nan():
    ok, _, _ = calibrate(lambda _sc: float("nan"))
    assert not ok


def test_linear_probe_acc_is_bounded_and_beats_chance():
    """准确率必须**有界**：R² 在多类 one-hot 上会发散到 −10⁹，准确率不会。"""
    torch.manual_seed(0)
    n, d, C = 2000, 32, 8
    Y = torch.randint(0, C, (n,))
    X = torch.zeros(n, d)
    X[torch.arange(n), Y] = 5.0  # 类别直接写进特征
    X += torch.randn(n, d) * 0.1
    acc = linear_probe_acc(X, Y, 1500, C)
    assert 0.0 <= acc <= 1.0
    assert acc > 1.0 / C + 0.3, f"可学的映射没被学到：{acc}"


def test_linear_probe_acc_at_chance_for_random_features():
    torch.manual_seed(0)
    n, d, C = 2000, 32, 8
    X = torch.randn(n, d)
    Y = torch.randint(0, C, (n,))
    acc = linear_probe_acc(X, Y, 1500, C)
    assert acc < 1.0 / C + 0.10, f"随机特征不该有高准确率：{acc}"


@pytest.mark.parametrize(
    "acc,firing,should_fail",
    [
        (0.60, 0.08, False),
        (float("nan"), 0.08, True),
        (0.60, float("inf"), True),
        (0.60, 0.001, True),  # 静默
        (0.60, 0.50, True),  # 饱和
        (0.13, 0.08, True),  # 底板（随机 0.125 + 0.15 门槛）
        (1.5, 0.08, True),  # 越界
    ],
)
def test_validity_gate(acc, firing, should_fail):
    ok, reasons = validity(acc, firing, chance=0.125)
    assert ok is (not should_fail), reasons


def test_assert_patched_raises_on_miss():
    with pytest.raises(RuntimeError):
        assert_patched("原文里没有这句", "要替换的字符串")


def test_assert_patched_passes_on_hit():
    assert_patched("这里有 old 片段", "old")
