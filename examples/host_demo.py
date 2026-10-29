"""50 行宿主 demo：inject / read_state / wait_trigger 三原语（SpikeBus 规范 v0.1）。

跑法：``python examples/host_demo.py``（无 GPU、无网络）。
"""

from __future__ import annotations

import numpy as np
from biosnn_bus import (
    BioSNNHost,
    ModalityPlugin,
    PassThroughMembrane,
    SpikeBus,
    SpikeTrain,
    TriggerEvent,
)


class SinePlugin(ModalityPlugin):
    @property
    def modality_name(self) -> str:
        return "sine"

    @property
    def spike_dim(self) -> int:
        return 32

    def encode(self, raw_input):
        return SpikeTrain(data=np.asarray(raw_input) > 0.5, channel=self.channel)

    def get_membrane(self):
        return PassThroughMembrane()

    def decode(self, spike_output):
        return spike_output.rates


def main() -> int:
    bus = SpikeBus(bus_dim=64, seed=0)
    bus.register(SinePlugin())
    fired = {"count": 0}

    def trigger() -> TriggerEvent | None:
        fired["count"] += 1
        return TriggerEvent(t=fired["count"], salience=0.9, associations=("demo",))

    host = BioSNNHost(bus, trigger=trigger)
    rng = np.random.default_rng(0)
    for _ in range(3):
        host.inject("sine", (rng.random((8, 32)) > 0.5).astype(np.float32))
    print("state:", host.read_state())
    event = host.wait_trigger(timeout_ms=10)
    print("trigger:", event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
