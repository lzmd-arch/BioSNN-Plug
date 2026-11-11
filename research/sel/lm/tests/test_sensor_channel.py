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
