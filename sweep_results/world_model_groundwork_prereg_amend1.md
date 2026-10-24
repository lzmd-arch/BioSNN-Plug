# 预注册修订 1：世界模型打底的终止头（2026-10-25）

**修订时间：2026-10-25。原预注册 [`world_model_groundwork_prereg.md`](world_model_groundwork_prereg.md)
不动；本文件只加一个基线档，阈值一个字不改。**

## 为什么要改

原运行（5 种子，见 [`world_model_groundwork/README.md`](world_model_groundwork/README.md)）的判定是
`task_not_ready`，但两半的失败方式完全不同：

- **状态**：`ridge_action` 的测试 NMSE 约 `1e-8`，`copy_last` 是 0.073——CartPole 的欧拉一步
  动力学本来就是 `[s, a, a·s]` 的线性函数，线性模型把它解掉了；
- **终止**：`ridge_action` 的 AUROC 只有 **0.53**，与常数分数（0.5）几乎没差。

原因已定位：**终止是双侧阈值**（`|θ'| > 0.2095` 或 `|x'| > 2.4`），单个线性分数
`w·[s,a]` 只能抬高一侧、压低另一侧，两侧贡献互相抵消。诊断（单种子、探索性，**不是**本修订的
结果）：

| 诊断项 | AUROC |
| :--- | ---: |
| 线性终止头（原预注册的 `ridge_action`） | 0.532 |
| 二次项终止头（`z²` 与 `(a·z)²`） | 0.997 |
| 由**预测出的** next state 直接按环境阈值推终止 | 1.000 |

所以原判据失败的**不是任务不可测**，是**那个基线表达不了双侧事件**。

## 修订内容

| 项 | 原预注册 | 本修订 |
| :--- | :--- | :--- |
| 主基线 | `ridge_action`（线性） | **`ridge_quadratic`**（线性项 + `z²` + `(a·z)²`，λ=1.0） |
| `ridge_action` | 主基线 | 降为**负对照**，仍然报 |
| 判据 | NMSE ≤ 0.90×`copy_last`；动作敏感度 ≥ 0.01；终止 AUROC ≥ 0.75 | **不变** |

`ridge_quadratic` 的终止头是**学出来的**，不使用环境阈值——用已知阈值去推终止是作弊，
只作为诊断项出现，不进判据。

## 运行

- 数据协议、切分、种子（0–4）与原预注册完全一致；
- 输出写**新目录** `sweep_results/world_model_groundwork/amend1/`，不覆盖原结果；
- 命令：`uv run python -m research.cognitive.run_world_model --seeds 0-4 --primary-model ridge_quadratic --out sweep_results/world_model_groundwork/amend1`

## 判据与失败处置

三条判据全过（5 个种子）→ `task_ready`；任一不过 → 仍判 `task_not_ready`，并说明是
状态、动作条件化还是终止哪一侧失败。**不因为"二次项看起来该有用"而调阈值。**
