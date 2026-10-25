"""W3 诊断 ③ 的守卫：bootstrap 可复现，评测噪声本身可复现。"""

from __future__ import annotations

import numpy as np
import torch

from research.rstdp_spiking.cartpole import train
from research.rstdp_spiking.diagnose import bootstrap_ci


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean() -> None:
    values = np.array([10.0, 12.0, 14.0, 16.0, 18.0])

    first = bootstrap_ci(values, seed=0)
    second = bootstrap_ci(values, seed=0)

    assert first == second
    assert first[0] <= values.mean() <= first[1]


def test_eval_noise_freezes_the_stream() -> None:
    """同一份权重、同一批评测种子 → 同一串评测分数（否则诊断 ③ 量的是噪声源本身）。"""
    kwargs = {
        "seed": 0,
        "episodes": 2,
        "eval_every": 2,
        "eval_episodes": 1,
        "n_features": 16,
        "sigma": 1.0,
        "steps": 5,
        "learning_rate": 1e-3,
        "baseline_decay": 0.05,
        "critic_kind": "td",
        "critic_rate": 1e-2,
        "gamma": 0.99,
        "warmup": 0,
        "device": torch.device("cpu"),
        "eval_noise_seeds": 2,
    }

    first = train(**kwargs)
    second = train(**kwargs)

    assert len(first.eval_noise) == 2
    assert first.eval_noise == second.eval_noise
