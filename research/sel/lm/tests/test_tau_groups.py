"""G2 守卫：异质膜 τ 的分组必须与轨迹的衰减口径一致，而且**输入权重真的在学**。

这两条各自都曾经是坏的（2026-11-10 修）：
① `tau_vec` 用连续块，`group_ids` 用交错下标 ⇒ 痕迹按错误的 τ 衰减；
② `w_in` 的更新读的是 `trace.eps_in`，而分组路径根本不维护它 ⇒ 输入层完全没学。
"""

from __future__ import annotations

import math

import torch

from research.sel.lm.model import TokenEProp

VOCAB, N_REC = 50, 12


def _make(**kwargs) -> TokenEProp:
    return TokenEProp(
        VOCAB,
        N_REC,
        fan_in=4,
        recurrence="none",
        generator=torch.Generator().manual_seed(0),
        device="cpu",
        **kwargs,
    )


def test_groups_are_contiguous_blocks_matching_tau_vec() -> None:
    model = _make(tau_groups=3)
    assert model.tau_vec is not None
    assert model.group_slices is not None
    for lo, hi in model.group_slices:
        block = model.tau_vec[lo:hi]
        assert torch.all(block == block[0]), "一个 τ 组必须是连续的一整块"
    # 组与组之间确实不同
    means = [float(model.tau_vec[lo:hi][0]) for lo, hi in model.group_slices]
    assert means == sorted(means) and means[0] < means[-1]


def test_group_alpha_matches_exp_minus_one_over_tau() -> None:
    model = _make(tau_groups=3)
    assert model.group_alphas is not None
    for alpha, (lo, _) in zip(model.group_alphas.tolist(), model.group_slices, strict=True):
        assert abs(alpha - math.exp(-1.0 / float(model.tau_vec[lo]))) < 1e-6


def test_input_weights_learn_in_every_group() -> None:
    model = _make(tau_groups=3)
    before = model.w_in.clone()
    tokens = torch.randint(0, VOCAB, (4, 10), generator=torch.Generator().manual_seed(1))
    model.run(tokens, model.new_trace(4), valid=None)
    assert model.group_slices is not None
    for lo, hi in model.group_slices:
        moved = (model.w_in[:, lo:hi] - before[:, lo:hi]).abs().sum()
        assert float(moved) > 0, f"第 [{lo},{hi}) 组的输入权重没动"


def test_homogeneous_path_has_no_groups() -> None:
    model = _make()
    assert model.tau_vec is None and model.group_slices is None
    trace = model.new_trace(4)
    assert trace.eps_in_groups is None
