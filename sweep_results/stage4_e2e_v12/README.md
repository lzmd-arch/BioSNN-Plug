# 阶段四端到端 v12：K=8 对角协方差门（2026-10-28）

预注册：[`stage4_e2e_v12_prereg.md`](../stage4_e2e_v12_prereg.md)。复用 v11 的三臂
（baseline / snapshot85 / longtau）；每臂同时报 **diag（主门）** 与 shrunk（对照）。
5 个种子。

## 结论

**snapshot85 + diag 与 longtau + diag 都 ready：`v12_ready=true`。K=8 的边界是协方差
估计，不是 WM 容量。**

| 臂 | 门 | WM R² | WM MSE | 覆盖 | 弃权 | AUROC | DPAR | 状态控制 | ready |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| baseline | diag | 0.8926 | 0.0781 | 0.8052 | 0.9132 | 0.9495 | 0.934 | 0.125 | ❌ |
| baseline | shrunk | 0.8926 | 0.0777 | 0.7740 | 0.9532 | 0.9677 | 0.934 | 0.125 | ❌ |
| **snapshot85** | **diag** | **0.9936** | **0.00480** | **0.7820** | **0.9292** | **0.9535** | 0.934 | 0.126 | ✅ |
| snapshot85 | shrunk | 0.9936 | 0.00483 | 0.7552 | 0.9768 | 0.9806 | 0.934 | 0.126 | ❌ |
| **longtau** | **diag** | **0.90145** | **0.06947** | **0.8052** | **0.9244** | **0.9559** | 0.934 | 0.122 | ✅ |
| longtau | shrunk | 0.90145 | 0.06956 | 0.7776 | 0.9600 | 0.9716 | 0.934 | 0.122 | ❌ |

## 读法

- v11 的三臂里，snapshot85/longtau 已把 WM 修好，但 shrunk 门的 coverage 只有
  0.7552/0.7776；
- 换成 **diag**（逐维方差，无 28×28 协方差）后，coverage 到 **0.7820 / 0.8052**，
  七项全过；
- **失败的是协方差估计，不是 WM/DPAR**：baseline diag 的 coverage 0.8052 也过了，只是
  baseline 的 R² 0.8926 仍 <0.90；
- 两个 ready 臂里，snapshot85 的 R² 更高（0.9936），longtau 的覆盖更宽（0.8052）；
  两者都报，不把其中一个说成“必须”。

## 逐种子（diag）

| seed | baseline R²/cov | snapshot R²/cov | longtau R²/cov |
| :--- | :--- | :--- | :--- |
| 0 | 0.889 / 0.808 | 0.994 / 0.776 | 0.900 / 0.790 |
| 1 | 0.890 / 0.800 | 0.993 / 0.792 | 0.899 / 0.854 |
| 2 | 0.897 / 0.802 | 0.993 / 0.788 | 0.910 / 0.796 |
| 3 | 0.896 / 0.816 | 0.994 / 0.804 | 0.901 / 0.794 |
| 4 | 0.891 / 0.800 | 0.994 / 0.750 | 0.897 / 0.792 |

snapshot85 的 coverage 均值 0.7820 刚过 0.78；seed4 是 0.750，说明**边界很薄**，不是
宽裕通过。longtau 的 0.8052 更稳，但 R² 只有 0.90145。

## 这个 ready 覆盖什么

- **覆盖**：K=8、s=3、σ=0.3；shared-state 双 bank；DPAR 每幕独立；状态控制 0.12；
- **不覆盖**：K>8；s=2（v10 的 K≤4）；s=1.5；信息丢失型 OOD（v6/v7 边界不变）；
- snapshot85 是 bank W 的 t=85 快照，不是最终状态；longtau 加固定 τ=400；
- 不写成阶段四整体验收。

## 边界

1. K=8 的覆盖通过依赖 **diag 门**；shrunk 仍不过（0.7552/0.7776）。这是门的选择，
   不是 WM 的变化。
2. snapshot85 的 coverage 0.7820 只比 0.78 高 0.002；seed4 0.750，单种子不稳。
3. longtau 的 R² 0.90145 只比 0.90 高 0.00145；也是薄边界。
4. K>8 未测；不要把 K=8 的 diag 结果外推到更大 K。
5. 5 个种子；单次运行 177.5 s；没有调覆盖率/λ/τ。

## 与 v10/v11 的关系

| 版本 | K=8 | 失败项 |
| :--- | :--- | :--- |
| v10 | baseline + shrunk | R² 0.89975、coverage 0.7532 |
| v11 | snapshot/longtau + shrunk | WM 修好，coverage 0.755/0.778 |
| **v12** | **snapshot/longtau + diag** | **coverage 0.782/0.805，七项全过** |

v10 的 s≥2/K≤4 结论不变；v12 只补 K=8 的 gate 选择与 WM 读出时间/慢 τ 的边界。

## Repro

```text
实验名称：阶段四端到端 v12
日期：2026-10-28 02:33:33 中国标准时间
Git commit：dbcf457f55cabc6275927dbc5fcd88fac1cbd7aa
Git 状态：干净
Python：3.12.14
操作系统 / 架构：Windows 11 / AMD64
硬件：NVIDIA GeForce RTX 5060，8,124 MB，sm_120
随机种子：0,1,2,3,4
依赖快照：uv.lock sha256=3f0b0a6b1d8631b4660c996348c286a9001ba2c3b832f2173b8f96e906e6384e
运行命令：run_e2e_delayed_recall_v12.py --seeds 0-4
耗时：177.5 s
备注：K=8 三臂 baseline/snapshot85/longtau；diag 主门，shrunk 对照
备注：s=3、sigma=0.3；阈值只在 calibration 上取；覆盖率 0.78-0.82
```
