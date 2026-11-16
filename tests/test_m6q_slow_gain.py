"""M6q 的守卫：`gain` 是**无泄漏积分器**，eta 只改累积速度。

预注册写"τ ≈ 4/eta"，那是开环估计；开环下它根本没有时间常数（无泄漏 ⇒ 记忆无界）。
这条测试钉住的是真正可断言的性质：**固定区间上的累积漂移 ∝ eta**。
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _integrate(eta: float, rate: float, steps: int) -> float:
    """按 `run_spont` 的更新规则积分，返回 log(gain) 的总变化。"""
    f = 0.08
    logg = 0.0
    for t in range(steps):
        f = 0.9 * f + 0.1 * rate
        if t % 4 == 3:
            logg += eta * (0.08 - f)
    return logg


def test_gain_drift_scales_with_eta() -> None:
    """eta 减半，同一区间上的漂移必须也减半（**线性**，不是 sqrt）。"""
    a = abs(_integrate(0.02, 0.20, 400))
    b = abs(_integrate(0.01, 0.20, 400))
    assert a != 0.0
    assert abs(a / b - 2.0) < 0.02, f"a/b = {a / b:.4f}，不是 2"


def test_gain_has_no_leak() -> None:
    """无泄漏 ⇒ 持续偏离会一直累积。若有泄漏，长区间上会饱和。"""
    short = _integrate(0.02, 0.20, 400)
    long = _integrate(0.02, 0.20, 20000)
    assert abs(long) > 5.0 * abs(short), f"短 {short:.4f} / 长 {long:.4f}：疑似有泄漏"


def test_slower_eta_is_slower() -> None:
    """阶梯必须真的是阶梯：eta 越大，同区间漂移越大。"""
    vals = [abs(_integrate(e, 0.20, 400)) for e in (0.2, 0.05, 0.02, 0.01, 0.005)]
    assert all(a > b for a, b in itertools.pairwise(vals)), vals


def test_clamp_bounds_exist() -> None:
    """`run_spont` 必须把 gain 夹在 [0.2, 5.0]，否则慢增益会漂到无穷。"""
    import inspect

    from scripts import m6_pools

    src = inspect.getsource(m6_pools.run_spont)
    assert "torch.clamp(gain" in src
    assert "0.2, 5.0" in src


def test_gain_route_all_scales_recurrence_too() -> None:
    """M6r 的接线臂：`gain_route="all"` 必须把增益乘到 `rec` 上，不止 `cur`。

    这是 M6r 唯一改的东西 —— 如果它被误删，两臂会逐位相同，主效应自动变 0 而没人发现。
    """
    import inspect

    from scripts import m6_pools

    src = inspect.getsource(m6_pools.run_spont)
    assert 'gain_route == "all"' in src
    assert src.count("rec = rec * gain.unsqueeze(0)") == 2, "历史期与自发期都该有"
