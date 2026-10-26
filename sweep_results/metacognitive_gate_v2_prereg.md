# 预注册：元认知门控 v2——读出 ensemble（2026-10-26）

**定稿时间：2026-10-26，在 v2 的任何数据之前。** v1 的 T-wide/margin 都把 AUROC 推到
**0.745**（门槛 0.75）、ECE 到 **0.132**（门槛 0.10）；差一点。v2 只换**置信度的来源**：
用 K 个独立读出的 ensemble，而不是单个读出的 softmax。

## 机制（ensemble 均值 + 温度）

```text
K=5 个 DeltaReadout(512→100)，每个在训练集的 80% bootstrap 子样本上训练（不同种子）
成员 logits 取平均 → 在校准集上拟合一个温度 T（网格 [0.01, 5]）
置信度 = softmax(mean_logits / T) 的最大值
阈值仍只在校准集上取 80% 覆盖
```

- 读出仍是 delta 规则，无反向传播；ensemble 只增加**训练样本/初始化的多样性**；
- 不做 stacking、不做学习型融合；T 仍是一个标量；
- 报告成员 AUROC 的均值与范围（ensemble 是否真的带来多样性，要看见）。

## 数据与协议

与 v0/v1 相同：CIFAR-100、冻结 ResNet18、5 个校准/评估切分、80% 覆盖、10 箱 ECE。
唯一变化：K=5 个 bootstrap 读出的均值 logits。

## 判据（与 v0/v1 相同，先写死）

1. 选择性准确率（80% 覆盖）比 base **≥ +5 pp**；
2. 正确性 AUROC **≥0.75**；
3. 温度缩放后 ECE **≤0.10**；
4. 三条全过 → `v2_ready`；否则报边界，不调 K/bootstrap/T 网格。

## 不做什么

- 不换特征/不调 η/epoch/切分；
- 不做 stacking/学习型融合/OOD/LLM 触发；
- 不因为某个成员好看就只报它；ensemble 与成员都报；
- 不在看到数据后改 K 或 bootstrap 比例。

## 产物

- `sweep_results/metacognitive_gate_v2/summary.json` + README（ensemble 与成员）。
