# Step 0 · R-delta minibatch（回放的主对照）

冻结 ResNet18 + delta 线性读出，minibatch **B=128**、ρ=0、20 epoch、5 种子；CIL 口径。

| ACC | AF |
| ---: | ---: |
| **0.2466**（0.2460–0.2469） | **0.8047**（0.8016–0.8066） |

回放臂只与这一行比；全批量那条只作历史参考。自证：commit `60211a7`，工作区干净。
总表见 [`../README.md`](../README.md)。
