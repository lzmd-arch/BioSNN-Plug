"""预注册 CIL 口径的守卫：样本按任务切，argmax 在**全部已见类**里。

同一份 logits，两种口径可以差很多——这正是连续出错两次的地方：

- 累计类集口径（原 `formal/`）：a[t][i] 变成"前 i+1 个任务的联合准确率"；
- oracle 口径（`formal_corrected/`）：argmax 只在本任务 10 类里，那是**上界**；
- 预注册口径：样本按任务切，分类器面对任务 t 时的全部已见类，**不提供任务 id**。
"""

from __future__ import annotations

import numpy as np
import torch

from research.continual.readout import evaluate
from research.continual.run_continual import evaluate_task


class _IdentityFeatures:
    def features(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _PrefersClassTwo:
    """无论输入是什么，都把 class 2 的 logit 打到最高。"""

    def predict(self, features: np.ndarray) -> np.ndarray:
        out = np.zeros((features.shape[0], 4), dtype=np.float64)
        out[:, 2] = 1.0
        return out


def test_cil_metric_keeps_all_seen_classes_in_the_argmax() -> None:
    x = torch.eye(4)
    y = torch.tensor([0, 1, 2, 3])
    task_a = np.array([0, 1])
    seen = np.array([0, 1, 2, 3])

    cil = evaluate_task(_IdentityFeatures(), _PrefersClassTwo(), x, y, task_a, seen)
    oracle = evaluate(_PrefersClassTwo(), x.numpy()[:2], y.numpy()[:2], task_a)

    assert cil == 0.0, "全部已见类的 argmax 选了 class 2，任务 A 的两个样本都错"
    assert oracle == 0.5, "oracle 只在本任务两类里选，样本 0 对、样本 1 错"


def test_cil_metric_returns_nan_when_the_task_has_no_samples() -> None:
    x = torch.eye(4)
    y = torch.tensor([0, 1, 2, 3])

    assert np.isnan(
        evaluate_task(
            _IdentityFeatures(),
            _PrefersClassTwo(),
            x,
            y,
            np.array([9]),
            np.array([0, 1, 2, 3]),
        )
    )
