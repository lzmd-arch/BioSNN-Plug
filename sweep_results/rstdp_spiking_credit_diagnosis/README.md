# 脉冲策略路径的诊断（2026-10-26）

这一组回答"修好痕迹、换掉信号之后，剩下的瓶颈在哪"。

- **D1（rate-features）**：[`rate_features/`](rate_features/README.md)。把资格换回速率版形状的
  `x·a`，10 种子中位数 62.15（低于 sign-consistent 75.65，3/10 变好）→ **不是资格形状的问题**。
- **D2（线性策略配方）**：[`../rstdp_spiking_policy_control/linear_rate/`](../rstdp_spiking_policy_control/linear_rate/README.md)。
  同一 harness 只换线性 Boltzmann 配方，中位数 **179.80**（相对 D1 配对 +124.40，10/10 变好）
  → **harness 能学**；瓶颈在脉冲计数选动作这条策略路径。

两者的共同读数：D1 的 `x·a` 是线性 logits 的梯度形状，而脉冲策略的动作由 LIF 计数决定；
资格与策略不同向。下一步是设计与**计数策略同向**的局部资格，而不是改用线性策略。

## 下一步：计数同向资格（预注册 v3）

按 D2 的判读，下一步不是改用线性策略，而是设计与计数策略同向的局部资格：
`E ← λE + x ⊗ [z ⊙ (1[a] − π)]`（`x` = 输入脉冲计数，`z` = LIF 发放计数，
`π` = 计数策略的实际采样分布）。预注册见
[`../rstdp_spiking_count_aligned_prereg.md`](../rstdp_spiking_count_aligned_prereg.md)；
**代码与数据尚未产生**。

