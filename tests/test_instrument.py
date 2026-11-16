"""仪器守卫的负向测试。

**为什么给检查写测试**：一个永远通过的检查比没有检查更糟——它给人一种已经守住了的错觉。
2026-11-16 一天之内，同一类工程错误让约 12 轮实验作废，所以这里**每条都先造出问题**，
断言它真的被抓到。
"""

from __future__ import annotations

# ruff: noqa: N806  —— 线性代数里 X/W/A/C 是惯例
import itertools
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.instrument import (
    FIRING_WINDOW,
    assert_patched,
    calibrate,
    linear_probe_acc,
    majority_baseline,
    probe_gain,
    scale_grid,
    seeded_dirichlet,
    validity,
)


def test_scale_grid_is_geometric_and_dense_enough():
    """网格必须够密：实测 ring 在 1.0→0.0027、2.0→0.311，粗网格会跨过窗口。"""
    g = scale_grid(0.4, 12.0, 1.15)
    assert g[0] <= 0.5 and g[-1] >= 10
    ratios = [b / a for a, b in itertools.pairwise(g)]
    assert all(abs(r - 1.15) < 1e-3 for r in ratios), "ratio 不是常数"


def test_calibrate_finds_a_valid_scale():
    """造一个只在 scale∈[1.1,1.4] 落进窗口的假网络。"""

    def fake(sc):
        return 0.08 if 1.1 <= sc <= 1.4 else (0.0 if sc < 1.1 else 0.6)

    ok, _sc, fr = calibrate(fake)
    assert ok and FIRING_WINDOW[0] <= fr <= FIRING_WINDOW[1]


def test_calibrate_rejects_silent_network():
    """**静默网络必须被判无效** —— 这是当天最贵的错误。"""
    ok, best, fr = calibrate(lambda _sc: 0.0001)
    assert not ok and best is None and math.isnan(fr)


def test_calibrate_rejects_nan():
    ok, _, _ = calibrate(lambda _sc: float("nan"))
    assert not ok


def test_linear_probe_acc_is_bounded_and_beats_chance():
    """准确率必须**有界**：R² 在多类 one-hot 上会发散到 −10⁹，准确率不会。"""
    torch.manual_seed(0)
    n, d, C = 2000, 32, 8
    Y = torch.randint(0, C, (n,))
    X = torch.zeros(n, d)
    X[torch.arange(n), Y] = 5.0  # 类别直接写进特征
    X += torch.randn(n, d) * 0.1
    acc = linear_probe_acc(X, Y, 1500, C)
    assert 0.0 <= acc <= 1.0
    assert acc > 1.0 / C + 0.3, f"可学的映射没被学到：{acc}"


def test_linear_probe_acc_at_chance_for_random_features():
    torch.manual_seed(0)
    n, d, C = 2000, 32, 8
    X = torch.randn(n, d)
    Y = torch.randint(0, C, (n,))
    acc = linear_probe_acc(X, Y, 1500, C)
    assert acc < 1.0 / C + 0.10, f"随机特征不该有高准确率：{acc}"


def test_majority_baseline_is_the_real_floor():
    """**底板必须实测**：类别 0 占 70% 时，1/8 = 0.125 会低得离谱。"""
    Ytr = torch.cat([torch.zeros(700, dtype=torch.long), torch.ones(300, dtype=torch.long)])
    Yte = torch.cat([torch.zeros(70, dtype=torch.long), torch.ones(30, dtype=torch.long)])
    floor = majority_baseline(Ytr, Yte, 2)
    assert abs(floor - 0.7) < 1e-6, floor
    assert floor > 1.0 / 2 + 0.19  # 与 1/n_class 的差距必须看得见


def test_majority_baseline_on_balanced_case_matches_chance():
    Ytr = torch.arange(8).repeat(100)
    Yte = torch.arange(8).repeat(10)
    assert abs(majority_baseline(Ytr, Yte, 8) - 0.125) < 1e-6


def test_probe_gain_is_normalised():
    assert abs(probe_gain(0.125, 0.125)) < 1e-9  # 在底板上
    assert abs(probe_gain(1.0, 0.125) - 1.0) < 1e-9  # 天花板
    assert abs(probe_gain(0.5625, 0.125) - 0.5) < 1e-9  # 走了一半


@pytest.mark.parametrize(
    "acc,firing,floor,n_test,should_fail",
    [
        (0.60, 0.08, 0.125, 1000, False),
        (float("nan"), 0.08, 0.125, 1000, True),
        (0.60, float("inf"), 0.125, 1000, True),
        (0.60, 0.001, 0.125, 1000, True),  # 静默
        (0.60, 0.50, 0.125, 1000, True),  # 饱和
        (0.376, 0.08, 0.376, 1000, True),  # **历史上的假结果**：acc 恰好等于多数类
        (1.5, 0.08, 0.125, 1000, True),  # 越界
        # 2026-11-16 的 64 类实战：1/64 随机线曾让门槛不可达；实测底板后这一档应当**通过**
        (0.057, 0.08, 0.025, 1440, False),
    ],
)
def test_validity_gate(acc, firing, floor, n_test, should_fail):
    ok, reasons = validity(acc, firing, floor, n_test)
    assert ok is (not should_fail), reasons


def test_validity_reports_every_problem_not_just_the_first():
    ok, reasons = validity(0.0, 0.5, 0.5, 100)
    assert not ok and len(reasons) >= 2, reasons


def test_assert_patched_raises_on_miss():
    with pytest.raises(RuntimeError):
        assert_patched("原文里没有这句", "要替换的字符串")


def test_assert_patched_passes_on_hit():
    assert_patched("这里有 old 片段", "old")


# ---------------------------------------------------------------- 任务播种
# 2026-11-16：`torch.distributions.Dirichlet(...).sample()` 走全局 RNG，不吃局部
# generator。四个脚本因此在**每次调用**都抽出一个新的 `pref`，于是训练集与测试集
# 是两个不同的任务，所有臂掉到随机 —— 却完全不像 bug。下面三条守死它。


def test_seeded_dirichlet_is_deterministic():
    a = seeded_dirichlet(3.0, 8, 0)
    b = seeded_dirichlet(3.0, 8, 0)
    assert torch.equal(a, b), "同一个 seed 必须给出同一个 pref"
    c = seeded_dirichlet(3.0, 8, 1)
    assert not torch.equal(a, c), "不同 seed 应当给出不同的 pref"


def test_seeded_dirichlet_does_not_disturb_global_rng():
    """`fork_rng` 的意义：仪器不能偷偷改动调用方的随机流。"""
    torch.manual_seed(123)
    before = torch.randn(3)
    torch.manual_seed(123)
    seeded_dirichlet(3.0, 8, 7)
    after = torch.randn(3)
    assert torch.equal(before, after)


def test_make_task_uses_one_pref_for_train_and_test():
    """**行为级守卫**：两个不同 seed 的 `make_task` 必须是同一个任务。"""
    from scripts.manifold_big import make_task

    K, V = 8, 64
    tr, st_tr = make_task(400, 32, K, 1, "cpu")
    te, st_te = make_task(400, 32, K, 2, "cpu")

    def prof(toks, st):
        out = torch.zeros(K, V)
        for s in range(K):
            out[s] = torch.nn.functional.one_hot(toks[st == s], V).float().mean(0)
        return out

    a, b = prof(tr, st_tr), prof(te, st_te)
    ca = (a - a.mean(1, keepdim=True)).flatten()
    cb = (b - b.mean(1, keepdim=True)).flatten()
    r = float((ca @ cb) / (ca.norm() * cb.norm() + 1e-9))
    assert r > 0.9, f"训练集与测试集不是同一个任务：corr={r:.3f}"


def test_no_script_draws_dirichlet_from_the_global_rng():
    """**静态守卫**：脚本里不许再出现未播种的 `Dirichlet(` 抽样。"""
    root = Path(__file__).resolve().parents[1] / "scripts"
    bad = []
    for f in sorted(root.glob("*.py")):
        if f.name == "instrument.py":  # 唯一允许出现的地方
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if "Dirichlet(" in line and "seeded_dirichlet" not in line:
                bad.append(f"{f.name}:{i}  {line.strip()}")
    assert not bad, "未播种的 Dirichlet：\n" + "\n".join(bad)


def test_mlp_probe_is_deterministic():
    """**探针必须播种**：同一次 `feats` 重跑两次要给同一个准确率。

    2026-11-16 发现 `dim_ladder.mlp` 原来没有 `manual_seed` —— 初始化与
    minibatch 顺序都随全局 RNG 漂，那个差被当成了臂与臂之间的差。
    """
    from scripts.dim_ladder import mlp

    torch.manual_seed(1234)
    X = torch.randn(600, 16)
    Y = torch.randint(0, 4, (600,))
    feats, st = X.unsqueeze(1), Y.unsqueeze(1)
    a = mlp(feats, st, 400)
    b = mlp(feats, st, 400)
    assert a == b, f"探针不可复现：{a} vs {b}"


def test_to_ball_does_not_saturate():
    """**双曲嵌入不许把所有点压到同一半径。**

    2026-11-16：标准化后的 4096 维特征模长约 64，`tanh(64)` 在 float32 下就是 1，
    于是 `to_ball` 把每个点都放到半径 0.95 的球面上。此时 Poincaré 距离是欧氏距离的
    严格单调函数，`argmin` 相同 —— 双曲分类器与欧氏分类器**逐位相同**，
    `manifold_big` 那次实验根本没测到双曲几何。症状：`poincare` 与 `euclid` 的
    逐种子数字一模一样。
    """
    from scripts.manifold_big import to_ball

    torch.manual_seed(0)
    for d in (16, 256, 4096):
        x = torch.randn(512, d)
        n = to_ball(x).norm(dim=1)
        assert float(n.std()) > 1e-3, (
            f"d={d} 时 to_ball 饱和：模长 {float(n.min())}~{float(n.max())}"
        )


def test_mlp_probe_accepts_3d_features():
    """`(序列, 步, 特征)` 也要能吃 —— `dim_ladder.mlp` 就是这个接口。

    2026-11-16：`mlp_probe` 只做了 `X[:cut]`，对 3-D 输入切的是**序列维**，
    于是 `mat1 and mat2 shapes cannot be multiplied (3072x8192 and 12x256)`。
    """
    from scripts.instrument import mlp_probe

    torch.manual_seed(0)
    X = torch.randn(80, 12, 32)
    Y = torch.randint(0, 8, (80, 12))
    acc3, _ = mlp_probe(X, Y, 64 * 12, 8, steps=20)
    acc2, _ = mlp_probe(X.reshape(-1, 32), Y.reshape(-1), 64 * 12, 8, steps=20)
    assert acc3 == acc2, f"3-D 与 2-D 输入结果不一致：{acc3} vs {acc2}"


def test_run_md_firing_is_bounded():
    """**发放率必须 ∈ [0,1]** —— 它是 0/1 值的均值。

    2026-11-16：`rate += z.sum() / alive.sum()` 里 `z` 是 `(b, n)`、
    `alive` 是 `(n,)`，分母少了一个 batch 倍数（1920），
    于是"发放"被报成 **10.3**。这一条守着它。
    """
    from scripts.memory_depth import run_md

    torch.manual_seed(0)
    dev = "cpu"
    N, D = 32, 4
    w_in = torch.randn(8, N) / N**0.5
    pre = torch.randint(0, N, (N, D))
    wrec = torch.rand(N, D)
    wrec = wrec * (1.2 / wrec.sum(1, keepdim=True))
    toks = torch.randint(0, 8, (4, 16))
    cue = torch.randint(0, 4, (4,))
    alive = (torch.rand(N) >= 0.5).float()
    for a in (None, alive):
        _, fr = run_md(w_in, pre, wrec, toks, cue, dev, 0.05, 0.004, 0.0, alive=a)
        assert 0.0 <= fr <= 1.0, f"发放率越界 {fr}（alive={a is not None}）"
