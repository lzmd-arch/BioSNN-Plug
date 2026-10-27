# 阶段四端到端 v15：只补 v14 的 4 个 s=2 失败格（2026-10-28）

预注册：[`stage4_e2e_v15_prereg.md`](../stage4_e2e_v15_prereg.md)。K4-σ0.5 主臂
longtau+shrunk；K8-s2 主臂 snapshot85+diag；5 种子。

## 结论

**四格全未 ready，`v15_ready=false`。** s=2 的弃权没有被换臂补上。

| 格 | 主臂 | 弃权 | 对照臂弃权 | 结果 |
| :--- | :--- | :--- | :--- | :--- |
| K4-s2-n0.5 | longtau | **0.7868** | baseline 0.7756 | ❌ |
| K8-s2-n0.1 | snapshot85 | **0.7700** | longtau 0.7656 | ❌ |
| K8-s2-n0.3 | snapshot85 | **0.7416** | longtau 0.7448 | ❌ |
| K8-s2-n0.5 | snapshot85 | **0.7044** | longtau 0.7024 | ❌ |

四格的 WM R²/MSE、DPAR 0.934–0.962、状态控制 0.12–0.25、AUROC 0.82–0.87 都过；
失败项只有弃权（门槛 0.80）。

## 读法

- K4-σ0.5 的 longtau 把弃权从 0.7756 抬到 0.7868，仍差 0.013；
- K8-s2 的 snapshot85 与 longtau 几乎相同（0.770 vs 0.766、0.742 vs 0.745、0.704 vs 0.702）；
- **换读出时间/慢 τ 不能解决 s=2 的弱尺度信号**；s=2 是明确边界，除非换更强的尺度特征或输入级门。

## 边界

1. v15 是 v14 之后的定向补格，不是新矩阵；
2. 失败项只有弃权；不能写成 WM/DPAR/状态失败；
3. 不调阈值/门/τ；s=2 仍记为边界；
4. 5 个种子；运行 649.3 s。

## Repro

```text
实验名称：阶段四端到端 v15
日期：2026-10-28 03:23:39 中国标准时间
Git commit：893367835b523968447e0b65bab526a82fabf6b0
Git 状态：干净
Python：3.12.14
操作系统 / 架构：Windows 11 / AMD64
硬件：NVIDIA GeForce RTX 5060，8,124 MB，sm_120
随机种子：0,1,2,3,4
依赖快照：uv.lock sha256=3f0b0a6b1d8631b4660c996348c286a9001ba2c3b832f2173b8f96e906e6384e
运行命令：run_e2e_delayed_recall_v15.py --seeds 0-4
耗时：649.3 s
备注：只跑 v14 的 4 个 s=2 失败格；K4-σ0.5 主臂 longtau，K8-s2 主臂 snapshot85
备注：门：K4 shrunk / K8 diag；覆盖率 0.78-0.82
```
