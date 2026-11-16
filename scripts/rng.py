"""随机源双入口：**实验播种，部署不可重放**。

## 为什么要有这个文件

M7（个体史）的判据要求两件互相冲突的东西：

| 判据 | 要求 |
| :--- | :--- |
| M7-3 历史相同 ⇒ 输出**逐位相同** | **可复现** |
| 部署的那个个体不能被重放 | **不可复现** |

⇒ 同一份代码必须有两个随机源入口。全局换掉任何一个都会坏掉另一半。

## 三档，不是两档（必须说准）

| 档 | 种子 | 能否被我们重放 | 用在哪 |
| :--- | :--- | :--- | :--- |
| `seeded` | 显式给出 | **能** | 实验、判据、M7-3 |
| `unrecorded` | `os.urandom(8)`，**不记录** | **不能**（种子丢了） | 部署的个体 |
| `quantum` | 外部 QRNG 设备 | 不能，且**原理上无隐变量** | 未接 |

**`unrecorded` 与 `quantum` 的区别要说清楚**：
`unrecorded` 仍然是**确定性的** —— 只要有人拿到那些 8 字节，整条轨迹就能重放。
它只是**对我们**不可重放。真正的量子源才没有隐变量。

对 M7 的用途，`unrecorded` **够用**：个体不可重放，因为它自己的种子没人保存。
但**不能说它"真随机"** —— 说"未记录的种子"。
"""

from __future__ import annotations

import os

import torch

MODE_SEEDED = "seeded"
MODE_UNRECORDED = "unrecorded"
MODES = (MODE_SEEDED, MODE_UNRECORDED)


def make_generator(seed: int | None = None, mode: str = MODE_SEEDED) -> torch.Generator:
    """按模式返回一个 `torch.Generator`。

    - `seeded`：必须给 `seed`，结果可复现；
    - `unrecorded`：从 `os.urandom` 取种子且**不返回**，调用方拿不到它。
    """
    if mode not in MODES:
        raise ValueError(f"未知模式 {mode!r}，可选 {MODES}")
    if mode == MODE_SEEDED:
        if seed is None:
            raise ValueError("seeded 模式必须给 seed")
        use = int(seed)
    else:
        use = int.from_bytes(os.urandom(8), "little") >> 1  # 保持非负
    g = torch.Generator()
    g.manual_seed(use)
    return g


def describe(mode: str) -> str:
    """给报告用的一句话说明。"""
    if mode == MODE_SEEDED:
        return "播种：可复现（实验/判据）"
    return "未记录的种子：对我们不可重放；**不是**量子随机，只是没人保存那 8 字节"
