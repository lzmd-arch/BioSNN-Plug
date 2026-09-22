"""W1：核化 IB-Hebbian 感知层（计划书 §3.3、§七 第一阶段）。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

验证的命题：**Pogodin & Latham (2020) 的核化 IB 导出的三因子 Hebbian 规则，加上
除法归一化，能否作为感知层特征提取器在 MNIST 上达到计划书 §七 要求的 70%+。**

模块划分：

* :mod:`~research.ib_hebbian.divisive_norm` —— Eq. (13)/(17) 的除法归一化；
* :mod:`~research.ib_hebbian.objective` —— Eq. (9)/(34)/(35) 的局部目标与二值教学信号；
* :mod:`~research.ib_hebbian.layers` —— 每层自带优化器的局部目标层；
* :mod:`~research.ib_hebbian.model` —— 多层堆叠 + 线性读出；
* :mod:`~research.ib_hebbian.train_mnist` —— 验收运行脚本。
"""
