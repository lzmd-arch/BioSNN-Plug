"""CIFAR-100 的加载与类增量任务划分。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

预注册：[`sweep_results/continual_cifar100_prereg.md`](../../sweep_results/continual_cifar100_prereg.md)。

## 为什么用 torchvision 下载

本仓库的 `scripts/download_data.py` 覆盖 MNIST 与 SHD，它们都是**裸文件 + 官方校验值**的形态。
CIFAR-100 是 tar 包，torchvision 自带的 CIFAR100 **对 tarball 做 MD5 校验**——那正是
`sweep_results/README.md` 里"校验值要有第三方来源"那条标准所要求的。所以这里用 torchvision
下载，而不是自己在 `download_data.py` 里再写一遍解析器。

⚠️ **数据不入库**：落在 `data/cifar100/`（`.gitignore` 已覆盖 `data/`）。

## 任务划分

100 类按**类别 id 升序**切成 10 组、每组 10 类。顺序固定 ⇒ 可复现，也便于与他人对照。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["CifarSplit", "load_cifar100", "task_classes"]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "data" / "cifar100"


@dataclass(frozen=True)
class CifarSplit:
    """一份 CIFAR-100 切分：``x`` 形状 ``(n, 32, 32, 3)`` 的 ``uint8``，``y`` 是 0–99。"""

    x: np.ndarray
    y: np.ndarray

    def subset(self, classes: np.ndarray, *, limit: int | None = None) -> CifarSplit:
        """只保留给定类别；``limit`` 从每类里取前若干个（用于冒烟）。"""
        mask = np.isin(self.y, classes)
        x, y = self.x[mask], self.y[mask]
        if limit is not None:
            keep = []
            for label in classes:
                idx = np.flatnonzero(y == label)[:limit]
                keep.append(idx)
            sel = np.concatenate(keep) if keep else np.array([], dtype=int)
            x, y = x[sel], y[sel]
        return CifarSplit(x=x, y=y)


def load_cifar100(
    root: Path | str = DEFAULT_ROOT, *, download: bool = True
) -> tuple[CifarSplit, CifarSplit]:
    """返回 ``(train, test)``。**下载由 torchvision 完成并校验 tarball 的 MD5。**"""
    from torchvision.datasets import CIFAR100

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    train = CIFAR100(str(root), train=True, download=download)
    test = CIFAR100(str(root), train=False, download=download)
    return (
        CifarSplit(x=np.asarray(train.data, dtype=np.uint8), y=np.asarray(train.targets)),
        CifarSplit(x=np.asarray(test.data, dtype=np.uint8), y=np.asarray(test.targets)),
    )


def task_classes(n_tasks: int = 10, n_classes: int = 100) -> list[np.ndarray]:
    """按类别 id 升序把 ``n_classes`` 切成 ``n_tasks`` 组，返回每组的类别数组。"""
    if n_tasks < 1 or n_classes < 1:
        raise ValueError(f"n_tasks 与 n_classes 必须为正，收到 {n_tasks}、{n_classes}。")
    if n_classes % n_tasks != 0:
        raise ValueError(f"类别数 {n_classes} 不能被任务数 {n_tasks} 整除——那不是均衡划分。")
    per = n_classes // n_tasks
    return [np.arange(i * per, (i + 1) * per) for i in range(n_tasks)]
