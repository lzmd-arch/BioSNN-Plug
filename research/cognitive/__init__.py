"""认知层组件（ADR-0015 / ADR-0016）。

唤醒通道是组件；世界模型目前只是**打底零件**——预注册见
``sweep_results/world_model_groundwork_prereg.md``，阶段四入口条件未变。
"""

from __future__ import annotations

from .arousal import ArousalChannel, ArousalConfig
from .world_model import (
    CopyLastModel,
    MeanDeltaModel,
    PerActionMeanModel,
    Prediction,
    RidgeActionModel,
    WorldModel,
    action_sensitivity,
    auroc,
    per_dim_nmse,
)

__all__ = [
    "ArousalChannel",
    "ArousalConfig",
    "CopyLastModel",
    "MeanDeltaModel",
    "PerActionMeanModel",
    "Prediction",
    "RidgeActionModel",
    "WorldModel",
    "action_sensitivity",
    "auroc",
    "per_dim_nmse",
]
