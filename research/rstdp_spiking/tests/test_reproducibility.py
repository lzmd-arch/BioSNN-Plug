"""可复现性守卫：同一个种子跑两遍，训练轨迹必须逐回合一致。

这条测试来自一次真实事故：训练回合 `env.reset(seed=None)` 用的是系统熵初始化的 RNG，
于是**同一个种子两次运行给出完全不同的轨迹**，而所有下游数字看起来都正常。
评测路径当时是有种子的，所以"评测可复现"这句话骗过了当时的检查。

⚠️ 这里只跑 2 个回合——测试要便宜，但**必须真的跑训练**，否则守卫是空的。
"""

from __future__ import annotations

import torch

from research.rstdp_spiking.cartpole import train


def _short_run():
    return train(
        seed=0,
        episodes=2,
        eval_every=2,
        eval_episodes=1,
        n_features=16,
        sigma=1.0,
        steps=5,
        learning_rate=1e-4,
        baseline_decay=0.05,
        warmup=0,
        critic_kind="td",
        critic_rate=1e-2,
        gamma=0.99,
        device=torch.device("cpu"),
    )


def test_same_seed_gives_identical_training():
    first = _short_run()
    second = _short_run()
    assert first.returns == second.returns, (
        "同一个种子两次训练的逐回合回报不同 —— 训练回合漏了环境种子？"
    )
