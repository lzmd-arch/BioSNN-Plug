# 预注册：P3 v2——用想象状态训练终止头（2026-10-27）

**定稿时间：2026-10-27，在本轮任何数据之前。** P3 的终止头在真实状态上 AUROC 0.999，
接回闭环只剩 **0.665**：闭环预测状态在约 115 步后发散，头看到的是训练分布外的状态。
v2 不改状态模型，只改**终止头的训练输入**：用模型的一步预测状态当特征，标签仍取真实
转移动的 `terminated`。

## 机制

```text
ridge_quadratic 学一步动力学（不变）
训练终止头：
  对每个训练转移 (s, a, terminated)：
      ŝ = s + ridge.delta(s, a)          # 模型想象的一步后状态
  用 (ŝ, a) 作输入、terminated 作标签，类平衡 L2 逻辑回归（IRLS）
闭环：状态增量仍 ridge；终止分数用该头；H=500
```

- 标签来自**真实**转移的终止事件，不是从想象状态伪造的；头只换了输入分布；
- 不把闭环 rollout 的标签喂回头（那会循环）；只用一步想象；
- 与 v1 的唯一差别是训练输入 `ŝ` vs `s`。

## 判据（与 P3 相同，先写死）

1. 闭环 H=10 NMSE / `copy_last` ≤0.90；
2. 回合级终止率 ≥1% 且闭环终止 AUROC **≥0.75**；
3. 中位首发散步 ≥10；
4. 三条全过 → `p3_ready`；否则报边界，不调状态模型/头/阈值。

## 不做什么

- 不改 ridge/λ/特征/H/策略/判据；
- 不用多步想象标签、不做闭环标签回流；
- 不把离线 AUROC 当闭环成绩；闭环必须重测。

## 产物

- `research/cognitive/run_world_model_closed_loop.py` 增加 `--termination imagined`；
- `sweep_results/world_model_closed_loop_imagined/summary.json` + README。
