"""第一阶段三条验证线共用的基础设施。

**⚠️ 破坏性变更免责期：本目录下的代码在头 12 个月内（至 2027-09）不做任何兼容性
承诺。** 依据见计划书 §12.5 与 [`docs/adr/ADR-0005`](../../docs/adr/ADR-0005-versioning-policy.md)。

要拿来用的东西请用 [`biosnn-bus`](../../packages/biosnn-bus/)，它遵循 semver。

这里放的是**没有领域知识、任何主题都用得上**的基建：

* :mod:`~research.common.seeding` —— 随机种子控制与登记；
* :mod:`~research.common.device` —— 设备选择与显存峰值测量；
* :mod:`~research.common.spike_metrics` —— 计划书 §9 的网络健康指标；
* :mod:`~research.common.provenance` —— §12.4 复现记录的生成。

口径不一致的指标比没有指标更糟：两条线各报一个"活跃神经元比例"，读者会以为它们
可比。所以这些定义只写一遍。

**有领域含义的零件不在这里**（例如 CartPole 的状态编码），在
:mod:`research.components`：那层可以依赖本包，本包不依赖它。依赖方向由
`tests/test_architecture.py` 检查。
"""
