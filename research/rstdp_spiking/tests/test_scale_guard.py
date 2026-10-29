"""D18 的守卫：给编码加 ``scale`` 之后，**默认路径必须逐位不变**。

本测试钉住两件事：

1. `_encode(..., scale=None)` 与历史写法 `encode_state(_state_tensor(...), ...)` 逐位相同；
2. 同一次小冒烟训练（seed=3、6 回合）的逐回合回报与评测值与**加参数之前**记录的参考值
   **完全相同**——参考值是改动前用 `git show HEAD:` 版本跑出来的，存在这里当基准。

第二条是防"改了个默认参数、把历史数字悄悄变了"的那种回归：只比"能不能跑"是看不出来的。
"""

from __future__ import annotations

import torch

from research.components.cartpole_encoding import encode_state, population_centers
from research.rstdp_spiking.cartpole import CARTPOLE_HALF_WIDTHS, _encode, _state_tensor, train

#: 2026-10-30、加 `scale` 参数**之前**的冒烟输出（seed=3、6 回合、16 特征、8 步窗口）。
REFERENCE_RETURNS = [17.0, 12.0, 23.0, 26.0, 21.0, 33.0]
REFERENCE_EVALS = [9.5, 11.5]


def test_encode_without_scale_is_bit_identical():
    centers = population_centers(
        16,
        half_widths=CARTPOLE_HALF_WIDTHS,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )
    raw = [0.1, -0.2, 0.05, 0.3]
    legacy = encode_state(_state_tensor(raw, centers.device), centers, 1.0)
    current = _encode(raw, centers, 1.0, None)
    assert torch.equal(legacy, current)


def test_scale_divides_the_state_per_axis():
    centers = population_centers(
        16,
        half_widths=(1.0, 1.0, 1.0, 1.0),
        generator=torch.Generator(device="cpu").manual_seed(0),
    )
    raw = [1.2, -0.6, 0.105, 1.5]
    scale = torch.tensor(CARTPOLE_HALF_WIDTHS)
    scaled = _encode(raw, centers, 0.35, scale)
    manual = encode_state(torch.tensor([0.5, -0.2, 0.5, 0.5]), centers, 0.35)
    assert torch.allclose(scaled, manual, atol=1e-6)


def test_default_training_path_reproduces_the_recorded_reference():
    """默认参数（scale=None）下的小冒烟必须与改动前逐位相同。"""
    result = train(
        seed=3,
        episodes=6,
        eval_every=3,
        eval_episodes=2,
        n_features=16,
        sigma=1.0,
        steps=8,
        learning_rate=1e-3,
        baseline_decay=0.05,
        critic_kind="pop",
        critic_rate=0.1,
        gamma=0.99,
        pop_post="gradient",
        warmup=2,
        device=torch.device("cpu"),
        trace_rule="count-aligned",
        spiking_temperature=True,
        logit_scale_start=0.1,
        logit_scale_end=0.02,
        keep_actor=True,
        actor_after="final",
    )
    assert [float(value) for value in result.returns] == REFERENCE_RETURNS
    assert [float(point["mean_steps"]) for point in result.eval_scores] == REFERENCE_EVALS
