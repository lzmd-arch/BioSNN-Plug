"""测试用的辅助构造。

放在独立模块而不是 ``conftest.py``，是为了让测试文件能显式 import——pytest 不鼓励
import conftest。目录名 ``biosnn-bus`` 含连字符、无法作为包名，因此靠根
``pyproject.toml`` 里的 ``[tool.pytest.ini_options] pythonpath`` 把它挂进
``sys.path``。
"""

from __future__ import annotations

import numpy as np
from biosnn_bus import ModalityPlugin, PassThroughMembrane, SpikeTrain


class ConstantPlugin(ModalityPlugin):
    """可配置的最小插件，用来测总线行为。"""

    def __init__(
        self,
        *,
        modality: str = "const",
        dim: int = 4,
        steps: int = 8,
        dt: float = 1.0,
        channel: str = "temporal",
        temporal_scale: float = 10.0,
    ) -> None:
        self._modality = modality
        self._dim = dim
        self._steps = steps
        self._dt = dt
        self._channel = channel
        self._scale = temporal_scale

    @property
    def modality_name(self) -> str:
        return self._modality

    @property
    def spike_dim(self) -> int:
        return self._dim

    @property
    def fusion_channel(self) -> str:
        return self._channel

    @property
    def temporal_scale(self) -> float:
        return self._scale

    def encode(self, raw_input) -> SpikeTrain:
        data = np.asarray(raw_input, dtype=bool).reshape(self._steps, self._dim)
        return SpikeTrain(data=data, dt=self._dt, channel=self.channel)

    def get_membrane(self) -> PassThroughMembrane:
        return PassThroughMembrane()

    def decode(self, spike_output: SpikeTrain) -> np.ndarray:
        return spike_output.rates


def make_pattern(steps: int = 8, dim: int = 4, *, seed: int = 0) -> np.ndarray:
    """固定种子的 0/1 脉冲模式。"""
    return np.random.default_rng(seed).random((steps, dim)) > 0.5
