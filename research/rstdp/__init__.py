"""W3：R-STDP 执行层 + TD-LTP Critic（计划书 §3.2、§七 第一阶段）。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

验证的命题：**带 TD-LTP Critic 的 R-STDP 能否在 CartPole 上稳定学到 200 步以上，且
成功偏移 < 10%σR。**

内置的 Critic 不是可选项。R-STDP 有结构性无监督偏差——成功偏移达 σR 的约 25% 就无法
学习，偏移 < −0.4σR 时性能甚至低于学习前（导致遗忘）[Frémaux et al., 2010]。该敏感性是
通用特性，调 STDP 窗口或换权重依赖模型都解决不了；唯一的结构性解法是**刺激特异性奖励
预测**，向 R-STDP 提供无偏信号。

模块划分：

* :mod:`~research.rstdp.td_ltp` —— Critic 的学习规则（Frémaux et al. 2013, Eq. 17）；
* :mod:`~research.rstdp.rstdp` —— Actor 的 R-STDP 规则与权重归一化；
* :mod:`~research.rstdp.cartpole` —— 环境封装、群体编码与训练循环；
* :mod:`~research.rstdp.measure_bias` —— 成功偏移 / σR 的测量；
* :mod:`~research.rstdp.capacity_probe` —— 编码的**线性容量**上限（否定性结果）；
* :mod:`~research.rstdp.signal_probe` —— **学习信号的质量**：Critic 准不准、δ 是不是优势
  估计、规则的方向对不对。
"""
