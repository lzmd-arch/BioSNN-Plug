"""第一阶段三条验证线共用的基础设施。

**⚠️ 破坏性变更免责期：本目录下的代码在头 12 个月内（至 2027-09）不做任何兼容性
承诺。** 依据见计划书 §12.5 与 [`docs/adr/ADR-0005`](../../docs/adr/ADR-0005-versioning-policy.md)。

要拿来用的东西请用 [`biosnn-bus`](../../packages/biosnn-bus/)，它遵循 semver。

这里放的是三条线**必须口径一致**的东西：

* :mod:`~research.common.seeding` —— 随机种子控制与登记；
* :mod:`~research.common.device` —— 设备选择与显存峰值测量；
* :mod:`~research.common.metrics` —— 计划书 §9 的网络健康指标；
* :mod:`~research.common.provenance` —— §12.4 复现记录的生成。

口径不一致的指标比没有指标更糟：两条线各报一个"活跃神经元比例"，读者会以为它们
可比。所以这些定义只写一遍。
"""
