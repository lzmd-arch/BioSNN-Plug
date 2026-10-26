# 预注册：多时间尺度工作记忆 v2——局部 delta 读出（2026-10-27）

**定稿时间：2026-10-27，在本轮任何数据之前。** v1 已 ready，但读出是 ridge 闭式（测量头）；
v2 把它换成**局部 delta 规则**，其余（固定 τ 的漏积分器、三线索、数据、对照）完全不变。

## 机制（局部读出）

```text
状态 h（τ=2..200，共 28 维）与 v1 相同
读出：ŷ = W h + b
更新：W ← W + η·(y − ŷ)hᵀ / batch，b ← b + η·(y − ŷ) / batch
      —— 只用读出端的误差与突触前状态，无闭式求解、无反向传播
```

- η=0.01、20 epoch、B=128（先写死；失败不调）；
- 单尺度对照与 v1 相同：fast(τ=2)、mid(τ=20)、slow(τ=200)，各只取一块 4 维；
- 任务/线索引子/噪声/训练测试规模与 v1 完全一致。

## 判据（先写死）

1. 多尺度 delta 读出：A、B、C 的 R² **都 ≥0.50**；
2. 快 τ=2 的 A **<0.30**；
3. fast/mid/slow 三个单尺度各自至少漏一个线索（R² <0.50）；
4. **不因局部化明显退化**：多尺度 delta 的三线索平均 R² ≥ v1 ridge 的平均 R² **−0.10**；
5. 四条全过 → `w2_ready`；否则报边界，不调 η/epoch/B/τ。

## 不做什么

- 不做脉冲 ALIF（v3）；不改 τ 列表/数据/判据；
- 不把 ridge 的成绩混进 delta 的列；
- 不在看到数据后调学习率或回合数；
- 不因为 delta 差一点就回退到闭式。

## 产物

- `research/cognitive/working_memory.py` 增加 `LinearDeltaReadout`；
- `research/cognitive/run_working_memory_v2.py`：delta 与 ridge 对照；
- `sweep_results/working_memory_v2/summary.json` + README。
