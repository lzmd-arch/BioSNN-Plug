"""SpikeBus 规范 v0.1 的 C1–C5 一致性测试（宿主契约）。

对应 ``packages/biosnn-bus/README.md`` 的「一致性测试」一节；改规范时先改这里。
"""

from __future__ import annotations

import numpy as np
import pytest
from biosnn_bus import (
    BioSNNHost,
    ModalityPlugin,
    PassThroughMembrane,
    SpikeBus,
    SpikeTrain,
    UnknownModalityError,
)


class _Sine(ModalityPlugin):
    @property
    def modality_name(self) -> str:
        return "sine"

    @property
    def spike_dim(self) -> int:
        return 16

    def encode(self, raw_input):
        return SpikeTrain(data=np.asarray(raw_input) > 0.5, channel=self.channel)

    def get_membrane(self):
        return PassThroughMembrane()

    def decode(self, spike_output):
        return spike_output.rates


def _host() -> BioSNNHost:
    bus = SpikeBus(bus_dim=32, seed=0)
    bus.register(_Sine())
    return BioSNNHost(bus)


def test_c1_step_returns_bus_dim():
    host = _host()
    host.inject("sine", np.zeros((4, 16), dtype=np.float32))
    assert host.read_state()["dim"] == 32
    out = host.bus.step({"sine": np.zeros((4, 16), dtype=np.float32)})
    assert tuple(out.data.shape)[-1:] == (32,)


def test_c2_unknown_modality_raises():
    host = _host()
    with pytest.raises(UnknownModalityError):
        host.inject("nope", np.zeros((4, 16), dtype=np.float32))


def test_c3_time_alignment():
    host = _host()
    host.inject("sine", np.zeros((4, 16), dtype=np.float32), t=0)
    host.inject("sine", np.zeros((4, 16), dtype=np.float32), t=1)
    with pytest.raises(ValueError):
        host.inject("sine", np.zeros((4, 16), dtype=np.float32), t=-1)
    assert host.read_state()["steps"] == 2


def test_c4_dimension_constant():
    host = _host()
    dims = set()
    for _ in range(3):
        host.inject("sine", np.zeros((4, 16), dtype=np.float32))
        dims.add(host.read_state()["dim"])
    assert dims == {32}


def test_c5_snapshot_is_read_only():
    host = _host()
    host.inject("sine", np.zeros((4, 16), dtype=np.float32))
    snapshot = host.read_state()
    snapshot["dim"] = 999
    snapshot["modalities"].append("hacked")
    again = host.read_state()
    assert again["dim"] == 32
    assert again["modalities"] == ["sine"]
