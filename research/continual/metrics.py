"""持续学习的指标：平均增量准确率与遗忘。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

预注册：[`sweep_results/continual_cifar100_prereg.md`](../../sweep_results/continual_cifar100_prereg.md)。

## 定义（写死在这里，与预注册一致）

设 ``a[t][i]`` = 学完任务 ``t`` 之后、在任务 ``i`` 的测试类上的准确率（``i ≤ t``）：

```text
ACC = mean_{t=0..T-1} ( mean_{i=0..t} a[t][i] )
AF  = mean_{i=0..T-2} ( max_{t=i..T-1} a[t][i] − a[T-1][i] )
```

``ACC`` 是"学完每一步时对见过的类的平均表现"；``AF`` 是**每个旧任务相对它自己历史最好成绩
的回退**的平均。

⚠️ **为什么两个都要报**：``ACC`` 的分母随任务数增长，旧任务的退化会被稀释——只看 ``ACC``
会让"学新忘旧"看起来还行。

⚠️ **单位**：``AF`` 用**百分点**（``0.05`` 表示回退 5 个百分点），与 ``ACC`` 同为比例。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ContinualScores", "accuracy_matrix_shape", "summarise"]


@dataclass(frozen=True)
class ContinualScores:
    """一次持续学习运行的汇总指标。"""

    acc: float
    """平均增量准确率（比例，0–1）。"""
    af: float
    """遗忘（比例，0–1；``nan`` 当 T < 2）。"""
    per_task_final: tuple[float, ...]
    """学完全部任务后，各任务的准确率（顺序为任务 0..T-1）。"""
    per_task_best: tuple[float, ...]
    """各任务在其历史上的最好准确率。"""


def accuracy_matrix_shape(n_tasks: int) -> tuple[int, int]:
    """准确率矩阵的形状，永远是方阵——未学到的格子用 ``nan`` 占位。"""
    if n_tasks < 1:
        raise ValueError(f"n_tasks 必须 >= 1，收到 {n_tasks}。")
    return (n_tasks, n_tasks)


def summarise(matrix) -> ContinualScores:
    """从 ``T×T`` 的准确率矩阵算 ACC 与 AF。

    Args:
        matrix: ``(T, T)``；``matrix[t][i]`` 是学完任务 ``t`` 后在任务 ``i`` 上的准确率。
            ``i > t`` 的格子（尚未学过）必须是 ``nan``。

    Raises:
        ValueError: 形状不是方阵、对角线以外出现 ``nan``，或 ``i > t`` 的格子不是 ``nan``。
    """
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"准确率矩阵必须是方阵 (T, T)，收到 {array.shape}。")
    n = array.shape[0]

    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    if not np.all(np.isnan(array[upper])):
        raise ValueError("i > t 的格子（尚未学过）必须是 nan——否则说明评测口径漏了。")
    if np.any(np.isnan(array[~upper])):
        raise ValueError("已学过的格子不能是 nan。")

    # ACC：每一步对"见过的类"的平均
    steps = [float(np.mean(array[t, : t + 1])) for t in range(n)]
    acc = float(np.mean(steps))

    final = array[n - 1, :n]
    best = np.array([np.nanmax(array[i:, i]) for i in range(n)])

    af = float(np.mean(best[: n - 1] - final[: n - 1])) if n > 1 else float("nan")
    return ContinualScores(
        acc=acc,
        af=af,
        per_task_final=tuple(float(x) for x in final),
        per_task_best=tuple(float(x) for x in best),
    )
