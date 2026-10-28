# 预注册：事件优先级/选择性时间步更新（阶段四，2026-10-29）

**性质：阶段四任务“只在值得算的时间步上做权重更新”的最小实现。** ESPP 的原文机制是
用输入活动/损失阈值选择时间步，实测只更新 18%–27% 的时间步；本实验把它落到现有
e-prop 训练循环上，先验证“跳过低活动时间步是否不掉精度”。

## 机制

```text
每个 batch 的输入脉冲活动 a_t = Σ_i x_{t,i}（T,batch）
选活动最高的 fraction（默认 0.30）个 (t,batch) 时间步
学习信号 L_j^t 在未选时间步置 0；资格痕迹仍按原递推累积
权重更新 = 原 e-prop 公式，但只由被选时间步的 L 贡献
```

- 不新增可学习参数；不改网络/任务/优化器；
- `selective_fraction=None`（默认）时逐位等于原路径；
- 选中比例为按 (t,batch) 元素计，记录实际比例。

## 协议

- sMNIST、n_rec=256、batch=64、30 epoch、lr_in=lr_rec=2e-3、seed 0（CPU）；
- 对照：同一 seed、同一配置、不选择性更新；
- 两臂都记录测试准确率、活跃比例、脉冲稀疏度、实际选中比例。

## 判据（先写死）

1. 实际选中比例在 **0.25–0.35**；
2. 选择性臂测试准确率 ≥ 配对基线 **−1.0 pp**；
3. 活跃比例 >60%（W2 的同一硬指标）；
4. 三条全过 → `event_priority_ready=true`；否则报边界，不调 fraction/epoch/lr。

## 边界

- 这是**选择性更新**的最小实现，不是 ESPP 规则本身；不把 18–27% 的原文数字当本实现结果；
- 只测一个任务/一个种子；不外推到 SHD/其他网络；
- 未选时间步的资格痕迹仍在累积，更新只是不让它们进入权重——这是“跳过更新”，不是“跳过计算”。

## 产物

- `research/eprop/eprop.py` 的 `selective_fraction`（默认 None，原路径不变）；
- `research/eprop/train_sequential.py` 的 CLI；
- `sweep_results/event_priority/` 两份 seed JSON + README。
