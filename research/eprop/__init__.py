"""W2：e-prop 认知层（计划书 §3.1、§七 第一阶段）。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

验证的命题：**e-prop（ALIF + 自适应阈值 + 资格痕迹）能否在顺序任务上跑通，
且活跃神经元比例 > 60%**——后者是计划书 §七 第一阶段的硬指标。

模块划分：

* :mod:`~research.eprop.neurons` —— LIF/ALIF 动力学、伪导数、脉冲函数；
* :mod:`~research.eprop.traces` —— 资格痕迹（Eq. 25）与指数滤波；
* :mod:`~research.eprop.eprop` —— 学习信号与权重更新（三因子规则）；
* :mod:`~research.eprop.tasks` —— sMNIST 与 SHD；
* :mod:`~research.eprop.train_sequential` —— 验收运行脚本；
* :mod:`~research.eprop.bptt_baseline` —— 代理梯度对照，用于量化差距报告。
"""
