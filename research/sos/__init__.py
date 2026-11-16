"""脉冲 SoS 的实验线（见 DESIGN.md）。

跨主题的调用方只能 import 包根（`from research.sos import ...`），
所以这里把外部真正用到的符号转出来 —— 规则见 `tests/test_architecture.py`
的 `dependency_violations`。
"""

from research.sos.grid import make_maze, step

__all__ = ["make_maze", "step"]
