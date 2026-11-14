"""时间编码：**信息落在时间轴上**，而不是幅度上。"""
from __future__ import annotations

import numpy as np


def test_stronger_channel_spikes_earlier():
    from research.sel.lm.spike_encoding import latencies

    lat = latencies([1.0, 0.5, 0.0], steps=8)
    assert lat[0] == 0            # 最强 ⇒ 第 0 步
    assert lat[2] is None         # ≤ 阈值 ⇒ 不发放
    assert 0 < lat[1] < 7


def test_frames_have_one_spike_per_channel():
    from research.sel.lm.spike_encoding import encode

    frames = encode([1.0, 0.5, 0.2], steps=6)
    assert frames.shape == (6, 3)
    assert int((frames > 0).sum()) == 3     # 每通道一帧，不重复


def test_order_carries_information():
    """两个向量**数值接近但顺序不同**时，编码必须不同——这正是速率编码做不到的。"""
    from research.sel.lm.spike_encoding import encode

    a = encode([0.9, 0.7, 0.5], steps=6)
    b = encode([0.5, 0.7, 0.9], steps=6)
    assert not np.array_equal(a, b)
    assert (a > 0).sum() == (b > 0).sum()   # 同样的"总量"，不同的**时间**分布
