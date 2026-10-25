"""CL 归一化臂的输入变换（预注册 v3）。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

预注册：[`sweep_results/continual_v2_normalization_prereg.md`](../../sweep_results/continual_v2_normalization_prereg.md)。

## 为什么把它单独成模块

归一化不是一个分支里的两行 `(x - mu) / sd`：它有三个必须被**测试**钉住的性质——

1. 统计量只由**任务 1** 的训练特征决定，之后冻结（否则旧类权重的输入表示会漂移，
   结果无法归因到归一化本身）；
2. 同一组统计量用于**所有任务**的训练与测试特征；
3. ``--normalize none`` 必须**逐位不变**（否则"基线"在实现层就换了）。

所以这里把"拟合/应用"做成一个小的变换协议，runner 只负责在任务 1 结束时 ``fit`` 一次、
之后一律 ``transform``。两个刻度：

* ``native``：逐维标准化，输出就是 ``(x - mu) / sd``；
* ``match-task1``：再把输出乘一个标量，使任务 1 训练特征的**平均行范数**回到原始值。
  这一档是控制臂——它保持整体梯度尺度，只留下"逐维等尺度"的几何变化。
"""

from __future__ import annotations

import hashlib
from typing import Any, Protocol

import numpy as np

__all__ = [
    "DEFAULT_EPS",
    "FeatureTransform",
    "IdentityTransform",
    "Task1Standardizer",
    "build_transform",
]

DEFAULT_EPS = 1e-6
"""方差下限。常数维（std = 0）会在这里被夹住，不会变成 inf/nan。"""


class FeatureTransform(Protocol):
    """输入特征的拟合/应用协议；runner 在任务 1 上 ``fit`` 一次，之后只 ``transform``。"""

    def fit(self, features: np.ndarray) -> Any: ...

    def transform(self, features: np.ndarray) -> np.ndarray: ...

    def digest(self) -> dict[str, Any]: ...


def _as_2d(features: np.ndarray, *, name: str = "features") -> np.ndarray:
    array = np.asarray(features)
    if array.ndim != 2:
        raise ValueError(f"{name} 必须是二维 (n, d)，收到 {array.shape}。")
    if array.shape[0] == 0:
        raise ValueError(f"{name} 不能为空（没有样本就算不出统计量）。")
    return array


def _digest(array: np.ndarray) -> str:
    """对 float64 连续内存取 SHA-256；同一个变换在不同机器上可逐位复核。"""
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


class IdentityTransform:
    """不做任何变换。``--normalize none`` 走这条，输出与输入逐位相同。"""

    def fit(self, features: np.ndarray) -> IdentityTransform:
        self.n_features = _as_2d(features).shape[1]
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        array = _as_2d(features)
        if array.shape[1] != self.n_features:
            raise ValueError(f"特征维度 {array.shape[1]} 与拟合时的 {self.n_features} 不一致。")
        return array

    def digest(self) -> dict[str, Any]:
        return {"kind": "identity", "n_features": self.n_features}


class Task1Standardizer:
    """逐维标准化；统计量只在 ``fit`` 的那批特征上算一次，之后冻结。

    Args:
        scale: ``native`` = 只标准化；``match-task1`` = 再乘一个标量，使拟合批的
            平均行范数回到原始值（控制臂，保持整体梯度尺度）。
        eps: 标准差下限；常数维被夹到它，避免除零。
    """

    def __init__(self, *, scale: str = "native", eps: float = DEFAULT_EPS) -> None:
        if scale not in ("native", "match-task1"):
            raise ValueError(f"scale 只能是 'native' 或 'match-task1'，收到 {scale!r}。")
        if eps <= 0:
            raise ValueError(f"eps 必须为正，收到 {eps}。")
        self.scale = scale
        self.eps = float(eps)

    def fit(self, features: np.ndarray) -> Task1Standardizer:
        array = np.asarray(_as_2d(features), dtype=np.float64)
        self.n_features = array.shape[1]
        self.mu = array.mean(axis=0)
        # 下限夹在 sd 上，transform 里不做第二次判断——否则拟合/应用两处的规则会漂移。
        self.sd = np.maximum(array.std(axis=0), self.eps)
        self.mean_row_norm_before = float(np.linalg.norm(array, axis=1).mean())
        standardized = (array - self.mu) / self.sd
        self.mean_row_norm_standardized = float(np.linalg.norm(standardized, axis=1).mean())
        self.rescale = 1.0
        if self.scale == "match-task1":
            self.rescale = self.mean_row_norm_before / max(
                self.mean_row_norm_standardized, self.eps
            )
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        array = np.asarray(_as_2d(features), dtype=np.float64)
        if array.shape[1] != self.n_features:
            raise ValueError(f"特征维度 {array.shape[1]} 与拟合时的 {self.n_features} 不一致。")
        out = (array - self.mu) / self.sd
        if self.rescale != 1.0:
            out = out * self.rescale
        return out

    def digest(self) -> dict[str, Any]:
        """可复核的快照摘要：统计量哈希 + 标量本身（标量很小，直接记）。"""
        return {
            "kind": "task1-standardize",
            "scale": self.scale,
            "n_features": self.n_features,
            "eps": self.eps,
            "mu_sha256": _digest(self.mu),
            "sd_sha256": _digest(self.sd),
            "mean_row_norm_before": self.mean_row_norm_before,
            "mean_row_norm_standardized": self.mean_row_norm_standardized,
            "rescale": self.rescale,
        }


def build_transform(kind: str, *, scale: str = "native") -> FeatureTransform:
    """CLI 的两个开关 → 变换实例；未知取值直接报错，不静默退化。"""
    if kind == "none":
        if scale != "native":
            raise ValueError("--normalize none 时 --normalize-scale 没有意义，必须为 native。")
        return IdentityTransform()
    if kind == "task1-standardize":
        return Task1Standardizer(scale=scale)
    raise ValueError(f"未知的 normalize 取值：{kind!r}。")
