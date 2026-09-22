"""研究代码（第一阶段起填充）。

**⚠️ 破坏性变更免责期：本目录下的代码在头 12 个月内（至 2027-09）不做任何兼容性
承诺。** 依据见计划书 §12.5 与 [`docs/adr/ADR-0005`](../docs/adr/ADR-0005-versioning-policy.md)。

要做成包而不是一堆散脚本，是为了让三条验证线共享 :mod:`research.common` 的口径
（种子、设备、指标、复现记录）。入口脚本一律用模块方式跑::

    uv run python -m research.<主题>.<脚本>

直接 ``python research/<主题>/<脚本>.py`` 也能跑，但那样 ``sys.path`` 上是脚本所在
目录而不是仓库根，``research.common`` 就 import 不到了。
"""
