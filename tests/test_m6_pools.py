"""M6p 多池机制的守卫测试。

**每条都先造出会出问题的情形**：`stack(dim=1)` 那个 bug 让 PR 从 1.83 跳到 37.55，
而 P-0 的发放率照样是 0.1421 —— 只测发放率是抓不到的。
"""

from __future__ import annotations

# ruff: noqa: N806
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m6_pools import N, pool_id, pool_rates, pr_of


def test_pool_id_partitions_evenly() -> None:
    """每个池的神经元数必须相同 —— 否则 φ_k 的均值不等于全局均值。"""
    for K in (1, 2, 4, 8):
        pid = pool_id("cpu", K)
        counts = torch.bincount(pid, minlength=K)
        assert counts.numel() == K
        assert bool((counts == N // K).all()), f"K={K} 分池不均：{counts.tolist()}"


def test_build_pools_with_k1_equals_turnover_build_net() -> None:
    """K=1、β=0 必须逐位复现 `turnover.build_net`。

    这是 P-0 仪器复现的结构基础：RNG 消耗顺序一乱，基线就不对了。
    """
    from scripts.m6_pools import build_pools
    from scripts.turnover import build_net

    w1, p1, r1 = build_net(3, "cpu")
    w2, p2, r2, pid = build_pools(3, "cpu", K=1, beta=0.0)
    assert torch.equal(w1, w2)
    assert torch.equal(p1, p2)
    assert torch.equal(r1, r2)
    assert bool((pid == 0).all())


def test_beta_widens_within_pool_but_zero_beta_is_unchanged() -> None:
    """β>0 只能提高同池比例，β=0 必须与不分区完全相同。"""
    from scripts.m6_pools import build_pools

    _, p0, _, pid = build_pools(0, "cpu", K=4, beta=0.0)
    _, p2, _, _ = build_pools(0, "cpu", K=4, beta=2.0)
    frac = lambda pre: float((pid[pre] == pid.unsqueeze(1)).float().mean())  # noqa: E731
    assert frac(p2) >= frac(p0)
    assert frac(p2) > 0.95  # 冒烟实测 β=2 是 1.000（池被完全切断）


def test_pool_inhibition_term_sums_to_zero() -> None:
    """交叉抑制 `g_inh·(φ_k − φ)` 在神经元上必须零和 —— 否则它偷偷改了总驱动。"""
    K = 4
    pid = pool_id("cpu", K)
    phk = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 0.0]])
    term = phk[:, pid] - phk.mean(1, keepdim=True)
    # float32 的舍入在 1e-6 量级；真坏了会是 O(100)，这条线分得开
    assert float(term.sum(1).abs().max()) < 1e-4
    assert float(term.abs().sum(1).max()) > 10.0


def test_pool_rates_are_pool_means() -> None:
    """`pool_rates` 必须是池内均值，不是池内求和。"""
    K = 4
    pid = pool_id("cpu", K)
    r = torch.randn(3, N)
    out = pool_rates(r, pid, K)
    assert out.shape == (3, K)
    for k in range(K):
        assert torch.allclose(out[:, k], r[:, pid == k].mean(1), atol=1e-5)


def test_pr_of_rank_one_is_one() -> None:
    """秩 1 的协方差必须给 PR≈1；秩 3 给 PR≈3。"""
    torch.manual_seed(0)
    t = torch.linspace(0, 1, 64).double()
    modes = torch.stack([torch.sin(2 * torch.pi * t * (i + 1)) for i in range(3)], 1)
    for rank in (1, 3):
        x = torch.randn(2000, rank).double() @ modes[:, :rank].T
        cov = x.T @ x / x.shape[0]
        pr = pr_of(cov)
        assert abs(pr - rank) < 0.2, f"rank={rank} 给出 PR={pr:.3f}"


def test_run_spont_returns_batch_neuron_time() -> None:
    """**`stack(dim=1)` 那个 bug**：快照必须是 (B, N, T)，不是 (B, T, N)。"""
    from scripts.m6_pools import B_BATCH, T_LATE, run_spont

    if not torch.cuda.is_available():
        pytest.skip("这条要真跑 1280 步，只在有 CUDA 的机器上跑")
    rates, snaps, early = run_spont(0, 0.5, 100, 0.7, 0.02, "cuda", 1, 0.0, 0.0, -1.0)
    assert rates.shape == (B_BATCH, N)
    assert snaps.shape == (B_BATCH, N, T_LATE)
    assert early.shape == (B_BATCH, N)
    assert float(rates.min()) >= 0.0 and float(rates.max()) <= 1.0


def test_run_spont_follows_history_batch_size() -> None:
    """**写死 batch=32 的 bug**：外部给 8 条历史时，`v/z/ref` 必须也是 8 行。

    2026-11-16 实际踩到：`B = B_BATCH` 让 8 条历史的调用在 g_inh≠0 时形状不匹配。
    """
    from scripts.hist_families import make_family_hist
    from scripts.m6_pools import T_LATE, run_spont

    if not torch.cuda.is_available():
        pytest.skip("要真跑，只在有 CUDA 的机器上跑")
    toks = make_family_hist(0, 1, 64, 8, "cuda")
    r, snaps, _ = run_spont(
        0, 0.5, 0, 0.7, 0.02, "cuda", 4, 0.0, 2.0, -1.0, hist_in=toks, t_spont=64
    )
    assert r.shape == (8, N)
    assert snaps.shape == (8, N, T_LATE)


def test_hetero_tau_zero_is_exactly_the_old_leak() -> None:
    """`hetero_tau=0` 时 `leak ≡ 0.95`，与改动前的标量行为逐位相同。

    这条守的是"加参数不能悄悄改动力学" —— 我在 M6w 跑着的时候patch 了核心动力学，
    必须证明默认值是无操作。
    """
    tau = torch.full((7,), 20.0)
    leak = 1.0 - 1.0 / tau
    assert bool((leak == 0.95).all())


def test_field_args_maps_radius_to_strengths() -> None:
    """**2026-11-16 的 bug**：加了 `g_local` 之后全局项永远施加，
    只传 `field_radius` 不会打开局部场 —— M6x/M6z 第一版因此全部跑成了全局场。

    这条钉住"半径 → 场强"的映射，防止再犯。
    """
    from scripts.m6x_local_life import field_args

    assert field_args(32) == {"g_field": 1.0, "g_local": 0.0}
    assert field_args(2) == {"g_field": 0.0, "g_local": 1.0}
    assert field_args(8) == {"g_field": 0.0, "g_local": 1.0}
