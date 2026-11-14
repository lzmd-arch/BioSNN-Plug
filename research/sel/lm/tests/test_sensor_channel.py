"""守卫：传感器通道**零初始化时与没有它逐位一致**。

这是 Semprea-4.5B 的承重件：传感器是"千问拿不到的信息"的唯一入口，但它
**今天没有配对数据**。所以它必须在"没有数据"时是**完全无副作用**的——
否则每一轮训练都会被一个恒为零的输入悄悄改变。
"""

from __future__ import annotations

import torch

from research.sel.lm.hierarchical import build_huffman_paths
from research.sel.lm.model import TokenEProp


def _build(sensor_dim: int) -> TokenEProp:
    return TokenEProp(
        16,
        32,
        recurrence="none",
        readout_taus=(2.0,),
        tree=build_huffman_paths(torch.ones(16)),
        input_std=1.2,  # 让神经元真的发放，否则守卫是空的
        sensor_dim=sensor_dim,
        generator=torch.Generator().manual_seed(0),
        device="cpu",
    )


def test_zero_sensor_is_bit_identical() -> None:
    plain = _build(0)
    with_sensor = _build(8)
    with_sensor.sensors = torch.zeros(8)
    tokens = torch.randint(0, 16, (4, 12))
    a = plain.run(tokens, plain.new_trace(4))
    b = with_sensor.run(tokens, with_sensor.new_trace(4))
    assert a["ce"] == b["ce"], (a["ce"], b["ce"])
    assert a["firing_rate"] == b["firing_rate"]


def test_no_sensors_attached_is_also_identical() -> None:
    """建了投影头但**没给向量**时也一样——两条"没有信息"的路径都必须无副作用。"""
    plain = _build(0)
    with_sensor = _build(8)
    tokens = torch.randint(0, 16, (4, 12))
    a = plain.run(tokens, plain.new_trace(4))
    b = with_sensor.run(tokens, with_sensor.new_trace(4))
    assert a["ce"] == b["ce"]


def test_nonzero_sensor_changes_output() -> None:
    """给了非零传感器、且投影头非零时，输出**必须变**——否则通道没接上。"""
    model = _build(8)
    with torch.no_grad():
        model.sensor_proj.normal_(0, 0.5)
    tokens = torch.randint(0, 16, (4, 12))
    base = model.run(tokens, model.new_trace(4))
    model.sensors = torch.randn(8) * 3.0
    fired = model.run(tokens, model.new_trace(4))
    assert fired["firing_rate"] != base["firing_rate"] or fired["ce"] != base["ce"]


# -- 多池 / 空步 / 侧抑制 / 有界：四道结构守卫 ---------------------------------


def _pooled(n_text: int = 8, n_pool: int = 4, n_extra: int = 4) -> TokenEProp:
    model = TokenEProp(
        16,
        n_text + n_pool + n_extra,
        recurrence="none",
        readout_taus=(2.0,),
        tree=build_huffman_paths(torch.ones(16)),
        input_std=1.2,
        sensor_dim=n_pool + n_extra,
        generator=torch.Generator().manual_seed(0),
        device="cpu",
    )
    with torch.no_grad():
        model.sensor_proj.normal_(0, 0.5)
    model.set_sensor_pools(
        {
            "a": (slice(0, n_pool), slice(n_text, n_text + n_pool)),
            "b": (
                slice(n_pool, n_pool + n_extra),
                slice(n_text + n_pool, n_text + n_pool + n_extra),
            ),
        }
    )
    return model


def test_pool_mask_blocks_cross_talk_after_update() -> None:
    """池结构必须**在更新之后**还成立——脚本里的掩码只管初始化，学一次就破。"""
    model = _pooled()
    model.sensors = torch.randn(8) * 3.0
    tokens = torch.randint(0, 16, (2, 6))
    model.run(tokens, model.new_trace(2))
    mask = model._pool_mask
    assert mask is not None
    assert float(model.sensor_proj.masked_select(~mask).abs().max()) == 0.0


def test_blank_step_has_no_text_current() -> None:
    """空步：**没有文本电流**。膜电位只走 α·v − z_prev·θ（无输入、无复发、无传感器）。"""
    model = _build(0)
    tokens = torch.randint(0, 16, (2, 6))
    tr = model.new_trace(2)
    model.step(tokens[:, 0], torch.zeros(2, dtype=torch.long), tr, update=False)
    v_before = tr.v.clone()
    z_prev = tr.z.clone()
    model.step(tokens[:, 1], torch.zeros(2, dtype=torch.long), tr, update=False, blank=True)
    expected = model.alpha * v_before - z_prev * model.threshold
    assert torch.allclose(tr.v, expected, atol=1e-6)
    # 同一 token 的非空步必须**不同**——否则说明 blank 没关掉输入
    tr2 = model.new_trace(2)
    model.step(tokens[:, 0], torch.zeros(2, dtype=torch.long), tr2, update=False)
    model.step(tokens[:, 1], torch.zeros(2, dtype=torch.long), tr2, update=False)
    assert not torch.allclose(tr2.v, tr.v, atol=1e-6)


def test_cross_inhibition_only_reads_other_pools() -> None:
    model = _pooled()
    z = torch.zeros(1, model.n_rec)
    z[0, 8:12] = 1.0  # 池 a 全发放
    out = z @ model.cross_inhib
    assert float(out[0, 8:12].abs().max()) == 0.0  # 池 a 自己不受影响
    assert float(out[0, 12:16].min()) > 0.0  # 池 b 被抑制


def test_sensor_clip_bounds_the_norm() -> None:
    model = _pooled()
    model.sensor_clip = 5.0
    model.sensors = torch.randn(8).abs() * 5.0
    tokens = torch.randint(0, 16, (2, 6))
    for _ in range(20):
        model.run(tokens, model.new_trace(2))
    assert model.sensor_norm <= 5.0 + 1e-4


# -- 训练器（多池任务）的两条守卫 -----------------------------------------------


def test_episode_label_follows_stimulus() -> None:
    """答案必须由**实际给的驱动**决定。

    `time` 池踩过一次：驱动器插了 g 个空步，但标签被另一条路径覆盖，
    于是"答案"和"激励"对不上，池被跑成随机（0.000 ↔ 1.000）。
    """
    import numpy as np

    from research.sel.lm.train_sensor_task import POOL, _make_episode

    rng = np.random.default_rng(0)
    for _ in range(20):
        ans, sensors, gaps = _make_episode("time", 8, rng, 0)
        assert gaps is not None
        assert (ans == gaps).all(), (ans, gaps)
        assert sensors.shape == (POOL, 8, 1)
        assert float(sensors.min()) == 1.0  # 张力性驱动是常数 1


def test_pools_are_disjoint() -> None:
    """五池的 (输入列, 神经元列) 必须互不相交，且不碰文本列。"""
    from research.sel.lm.train_sensor_task import POOL, POOLS, TEXT_N, _pool_columns

    seen_in: set[int] = set()
    seen_col: set[int] = set()
    for k, name in enumerate(POOLS):
        i_sl, c_sl = _pool_columns(name, k)
        assert c_sl.start >= TEXT_N
        for i in range(i_sl.start, i_sl.stop):
            assert i not in seen_in
            seen_in.add(i)
        for j in range(c_sl.start, c_sl.stop):
            assert j not in seen_col
            seen_col.add(j)
    assert len(seen_in) == POOL * len(POOLS)
    assert len(seen_col) == POOL * len(POOLS)
