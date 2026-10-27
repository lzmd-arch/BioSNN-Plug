# 阶段四端到端 v11：K=8 的容量/探针时间边界（2026-10-28）

预注册：[`stage4_e2e_v11_prereg.md`](../stage4_e2e_v11_prereg.md)。三臂都跑：
baseline（bank W t=125）、snapshot85（bank W t=85）、longtau（t=125 + tau=400）；
bank D、DPAR、状态控制不变。5 个种子。

## 结论

**三臂都未 ready，且失败项不是 WM 容量，而是 K=8 的 OOD 覆盖率。**

| 臂 | WM R² | WM MSE | 覆盖 | 弃权 | AUROC | DPAR | 状态控制 | ready |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| baseline | 0.8926 | 0.0777 | **0.7740** | 0.9532 | 0.9677 | 0.9336 | 0.1252 | ❌ |
| snapshot85 | **0.9936** | **0.00483** | **0.7552** | 0.9768 | 0.9806 | 0.9336 | 0.1256 | ❌ |
| longtau | **0.90145** | 0.06956 | **0.7776** | 0.9600 | 0.9716 | 0.9336 | 0.1216 | ❌ |

三个臂的 OOD 弃权、AUROC、DPAR、状态控制都过线；**唯一失败项是 test coverage 低于
0.78**（baseline 0.7740、snapshot85 0.7552、longtau 0.7776）。

## 读法

- **snapshot85** 把 WM R² 从 0.893 拉到 **0.9936**、MSE 0.005，说明 K8 的 WM 退化主要不是
  状态容量，而是 **t=125 的读出时间**；
- **longtau** 加 τ=400 后 R² **0.90145**（刚好过 0.90），MSE 0.0696，说明慢 τ 也能补容量；
- 但两臂的 coverage 仍然只有 0.755 / 0.778；**K8 的剩余边界转移到尺度门的校准/覆盖率**，
  不是 WM 读出本身；
- baseline 的 0.8926 比 v10 的 0.89975 略低（种子/实现差异），但同样 coverage 0.774。

## 为什么 coverage 会掉

K=8 的 bank D 时间线更长（query 在 t=120），bank W 仍是 A/B/Q；OOD 只在 A/B 上。K=8 的
状态控制/DPAR 都没问题，说明不是状态坏了；coverage 掉是**校准集与测试集在 56 维拼接状态
上的尾部行为**，不是 WM 或 DPAR 的排序失败。按预注册不调覆盖率/λ/窗口。

## 判据

预注册七项：R²≥0.90、MSE≤0.10、DPAR≥0.80、状态控制≤0.225、AUROC≥0.75、
弃权≥0.80、coverage 0.78–0.82。三臂都只挂 coverage：
`v11_ready=false`，`baseline/snapshot85/longtau` 均未 ready。

## 边界

1. snapshot85 是 bank W 的快照，不是共享最终状态；它证明“时间”假设，不证明 K8 共享状态
   已解决。
2. longtau 只加一个固定 τ=400；不调 τ 集合。
3. K=8 的 coverage 失败是**门校准**问题，不能写成 WM 容量边界；反过来，也不能因为
   R² 过 0.90 就追认 K8 ready。
4. DPAR 仍是每幕独立建库；OOD 仍是 s=3 尺度偏移。
5. 5 个种子；单次运行 204.5 s；没有调阈值/覆盖率/λ。

## 下一步

- 若闭合 K8：需要 **K=8 专用的校准/覆盖率修正**（例如按状态维度/时间窗分层校准），
  而不是继续加 τ 或调 R² 门槛；
- 若接受 K≤4：v10 的 s≥2、K≤4、σ≤0.5 仍是当前最干净的共享状态结论；
- 技术报告 v1.0 是 v10/v11 之前的快照；v11 的 K8 coverage 边界应写入下一版报告。

## Repro

```text
实验名称：阶段四端到端 v11
日期：2026-10-28 02:19:40 中国标准时间
Git commit：cd31823802bff0715398884ff750448c9ce53d9b
Git 状态：干净
Python：3.12.14
操作系统 / 架构：Windows 11 / AMD64
硬件：NVIDIA GeForce RTX 5060，8,124 MB，sm_120
随机种子：0,1,2,3,4
依赖快照：uv.lock sha256=3f0b0a6b1d8631b4660c996348c286a9001ba2c3b832f2173b8f96e906e6384e
运行命令：run_e2e_delayed_recall_v11.py --seeds 0-4
耗时：204.5 s
备注：K=8；三臂 baseline/snapshot85/longtau；bank D 与 DPAR 不变
备注：主门 shrunk Mahalanobis（lambda=0.1）；覆盖率判据 0.78-0.82
备注：s=3、sigma=0.3；阈值只在 calibration 上取
```
