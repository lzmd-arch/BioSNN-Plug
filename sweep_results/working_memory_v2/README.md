# 多时间尺度工作记忆 v2：局部 delta 读出（2026-10-27）

**预注册：** [`../working_memory_v2_prereg.md`](../working_memory_v2_prereg.md)。
状态/任务/线索与 v1 相同；读出从 ridge 换成局部 delta（η=0.01、20 epoch、B=128）；
2000 训练 / 500 测试、5 种子；干净工作区 commit `a772033`。

## 结果

| 臂 | A | B | C |
| :--- | ---: | ---: | ---: |
| delta multi | **0.031** | **0.022** | **0.040** |
| ridge multi（对照） | 0.994 | 0.988 | 0.998 |
| delta fast | −0.005 | −0.003 | −0.006 |
| delta mid | 0.001 | 0.002 | 0.022 |
| delta slow | 0.000 | −0.001 | 0.000 |

- `multi_ok=false`；`no_degradation=false`（delta 均值 0.031 vs ridge 0.993）；
- 单尺度对照仍全漏，但多尺度 delta 本身没学起来；`w2_ready=false`。

## 读数：局部 delta 在预注册步长下**严重欠拟合**

- 状态是 28 维漏积分器输出，量级不小；20 epoch × 2000 样本 = 320 次 minibatch 更新，
  η=0.01 下权重只走了一点点——delta 读出基本停在零附近（R² ~0.03）；
- 这不是"局部规则不可能"，而是**步长/更新次数与状态尺度不匹配**；按预注册不调 η/epoch/B，
  本轮记为不 ready；
- ridge 仍 0.993，说明状态表示没有问题（与 v1 一致）。

## 结论

**v2 不 ready；"局部学习的工作记忆"尚未证明。** v3 的自然方向是换一个**局部、尺度无关**
的读出规则：**NLMS**（`W ← W + η·error·x / (‖x‖²+eps)`）或 **RLS**（rank-one 局部更新），
而不是把 η 往上调——后者是调参，不是机制。

## 自证

`summary.json` 复现块：commit `a772033`，工作区干净。
