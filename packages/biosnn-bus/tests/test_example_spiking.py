"""示例 `examples/spiking_membrane.py` 必须真的能跑——它是"总线 → 膜 → 脉冲"的唯一完整路径。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "spiking_membrane.py"


def _load():
    spec = importlib.util.spec_from_file_location("spiking_membrane_example", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_example_produces_spikes_end_to_end():
    result = _load().main()
    assert result["steps"] == 32 and result["neurons"] == 64
    assert result["spikes"] > 0, "整段没有发放，说明总线与膜没接上"
    assert result["active_neurons"] > 0
