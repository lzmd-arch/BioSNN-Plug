# 预注册：液态 LIF 探针（Liquid-SNN，2026-10-28）

**性质：探索项探针，按桌面简报 `liquid_snn_agent_brief.md` 执行；不改变阶段四验收。**
**定稿时间：2026-10-28，在本轮任何液态 LIF 数据之前。** 完全复用 W2 的 sMNIST/e-prop
协议（250 epoch、batch 64、n_rec 256、lr 1e-2、LIF beta=0、threshold 0.62），只把
固定 τ 换成**由局部发放率现算的液态 τ**。

## 机制（唯一改动）

```text
标准 LIF： α = exp(−1/τ)，τ 固定 20
液态 LIF： r_i(t) = γ·r_i(t−1) + (1−γ)·z_i(t)        ← 本神经元自己的脉冲
          τ_i(t) = 1 / (a + b·r_i(t))，clamp 到 [5, 20]
          α_i(t) = exp(−1/τ_i(t))
          v_i(t) = α_i(t)·v_i(t−1) + I_i(t) − thr·z_i(t−1)
```

- 常数（预注册、不训练）：`a = 1/20`、`b = 0.15`、`γ = 0.9`；τ 范围 **[5, 20]**。
  r∈[0,1] 时 τ 从 20 降到 5；活动越强、记忆越短。
- `a,b,γ` 与 τ 计算**不在任何参数/优化器/`requires_grad` 路径上**；AST 测试钉死。
- e-prop 的 ε_v 递推使用**当步的 α_i(t)**；α 轨迹在反传（痕迹）里视为外部系数。
  **边界**：这忽略了 `∂α/∂z` 的直接项，因此本轮是“冻结液态 τ 轨迹的 e-prop”，不是
  对液态递推的精确梯度。实现必须附一个小规模 **frozen-α 梯度对照**（把 α 冻结后，
  e-prop 与 autograd 的梯度一致），证明痕迹公式本身没写错；`∂α/∂z` 的缺口在报告里
  明写，不藏。

## 协议（完全复用 W2）

- 数据：sMNIST，`load_smnist` 的相同预处理与脉冲编码；种子同时决定编码。
- 训练：e-prop 在线，250 epoch，5 种子（0–4），lr_in=lr_rec=1e-2，lr_out=5e-2，
  decay_out=0.95，batch 64，n_rec 256，threshold 0.62，beta=0（LIF）。
- 对照（直接引用，不重跑）：W2 LIF **0.8889**（种子 0–2、250 ep、lr 1e-2）、
  ALIF **0.8831**、BPTT 上界 **0.9630**。
- 硬件：RTX 5060 8GB；单种子 W2 实测约 5.9 h，5 种子约 30 h；冒烟只跑 1 epoch。

## 判据（先写死）

1. **主判据**：液态 LIF 五种子均值 > **0.8889**，且五种子 95% CI 下界 > 0.8889；
2. **次判据**：均值 > **0.8831**（ALIF 对照）；
3. 活跃神经元比例 > 60%（W2 的 §七 硬指标，必须同报）；
4. 五条（主判据 + 次判据 + 活跃比例）全过 → `liquid_lif_ready=true`；否则记负结果，
   不调 a/b/γ/τ 范围/判据；按简报 §6 结项归档，不追加轮次。

## 假说（先写死）

sMNIST 是规整网格，液态 τ 的增益预期小；若主判据不过，按简报记负结果并归档。
只有主判据通过，才预注册第二轮**事件流数据**（DVS/耳蜗）探针，不在本轮外推。

## 不做

- 不引入 ncps/CfC、不实现 FPTT（文献未核验）、不做 BPTT 训练；
- 不改 SpiceBus/认知核心/ADR；不把 DA-LIF/PLIF 的可学习 τ 放进研究线；
- 不把冻结 α 的 e-prop 写成精确液态梯度；不隐藏 `∂α/∂z` 缺口。

## 产物

- `research/eprop/liquid.py`（LiquidLIFCell + 局部 α 记录）；
- `research/eprop/eprop.py` 的 α 轨迹接口（标量路径保持原位）；
- `research/eprop/train_liquid.py`（复用 W2 训练循环）；
- `research/eprop/tests/test_liquid.py`（AST 守卫 + frozen-α 梯度对照 + 冒烟）；
- `sweep_results/liquid_lif/` 逐种子 JSON + README。
