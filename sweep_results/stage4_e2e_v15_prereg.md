# 预注册：阶段四端到端 v15——只补 v14 的 4 个 s=2 失败格（2026-10-28）

**定稿时间：2026-10-28，在本轮任何数据之前。** v14 的 18 格中 14 格 ready；失败只剩
s=2 的四格：K4-s2-n0.5（弃权 0.7756）、K8-s2-n0.1/0.3/0.5（0.7656/0.7448/0.7024）。
本预注册只跑这四格，并预先指定每格的**新臂**：

| 格 | 新臂（主） | 对照 |
| :--- | :--- | :--- |
| K4-s2-n0.5 | **longtau + shrunk**（32 维，τ=400） | baseline + shrunk（v14） |
| K8-s2-n0.1 | **snapshot85 + diag**（t=85 的 28 维） | longtau + diag（v14） |
| K8-s2-n0.3 | **snapshot85 + diag** | longtau + diag |
| K8-s2-n0.5 | **snapshot85 + diag** | longtau + diag |

## 判据（每格七项，5 种子均值）

1. WM R² ≥0.90；
2. WM 选择性 MSE ≤0.10；
3. DPAR 最好臂 ≥0.80；
4. 状态控制 ≤1/K + 0.10；
5. OOD AUROC ≥0.75；
6. OOD 弃权 ≥0.80；
7. test 覆盖率 0.78–0.82。

`v15_ready =` 四格的主臂全部七项过。对照臂只报告，不参与 `v15_ready`。

## 解释规则

- K8 s2 用 snapshot85 过 → s=2 的失败是 bank W 的**读出时间**，不是 diag 门；
- K4 s2 n0.5 用 longtau 过 → 是慢 τ 不足；
- 任一格主臂不过 → 该格仍记 s=2 边界；不调阈值/门/τ；
- 这是 **v14 之后的定向补格**，不是新的全矩阵；不能把 v15 的通过写成整张矩阵重新验证过。

## 不做什么

- 不重跑已过的 14 格；
- 不调 s/σ/K、阈值、覆盖率、λ；
- 不把 s=2 的通过外推到 s=1.5 或信息丢失型 OOD。

## 产物

- `research/cognitive/run_e2e_delayed_recall_v15.py`；
- `sweep_results/stage4_e2e_v15/summary.json` + README。
