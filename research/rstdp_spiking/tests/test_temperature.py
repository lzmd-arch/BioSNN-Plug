"""D6：脉冲 Actor 的温度调度开关。默认关闭时必须保持历史行为。"""

from __future__ import annotations

import pytest
import torch

from research.rstdp_spiking.cartpole import train


def _run(spiking_temperature: bool):
    return train(
        seed=0,
        episodes=2,
        eval_every=25,
        eval_episodes=1,
        n_features=8,
        sigma=1.0,
        steps=3,
        learning_rate=1e-3,
        baseline_decay=0.05,
        critic_kind="ema",
        critic_rate=0.1,
        gamma=0.99,
        warmup=10,
        device=torch.device("cpu"),
        trace_rule="count-aligned",
        keep_actor=True,
        logit_scale_start=1.0,
        logit_scale_end=0.01,
        spiking_temperature=spiking_temperature,
    )


def test_default_keeps_constant_scale():
    result = _run(False)
    assert result.actor.logit_scale == pytest.approx(1.0)


def test_enabled_applies_schedule():
    result = _run(True)
    # episodes=2：第二个回合 progress=0.5 → sqrt(1.0 × 0.01) = 0.1
    assert result.actor.logit_scale == pytest.approx(0.1, rel=1e-6)


def test_cli_accepts_new_flags(tmp_path):
    """CLI 必须真的注册这三个开关（曾经漏加，,--help 不报错但运行即崩）。"""
    from research.rstdp_spiking.cartpole import main

    code = main(
        [
            "--smoke",
            "--spiking-temperature",
            "--entropy-log",
            "--probe-every",
            "5",
            "--logit-scale-start",
            "0.1",
            "--logit-scale-end",
            "0.02",
            "--device",
            "cpu",
            "--out",
            str(tmp_path / "smoke"),
        ]
    )
    assert code == 0
