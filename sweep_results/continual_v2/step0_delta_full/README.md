# Step 0 · R-delta 全批量（v1 协议参考）

冻结 ResNet18 + delta 线性读出，**全批量**、20 epoch、5 种子；CIL 口径。

| ACC | AF |
| ---: | ---: |
| 0.2358（5 种子同值） | 0.7933 |

只作与历史协议对齐的参考，**不用它判回放成败**（回放的主对照是同 batching 的
R-delta-mini）。自证：commit `60211a7`，工作区干净。总表见 [`../README.md`](../README.md)。
