"""仪器：**先把"能不能测"这件事做成可复用的函数**，再谈结论。

## 为什么要有这个文件

2026-11-16 一天之内，同一类工程错误让约 12 轮实验作废，其中两次还跑出了
"看起来通过"的假结果（R² = −1.6×10⁹ 被判过、准确率 0.000 被当结论）。

四道闸，全部集中在这里，**任何实验都必须先过它们**：

| 闸 | 拦什么 |
| :--- | :--- |
| `calibrate` | 网络静默 / 饱和；网格跨过窄窗口 |
| `linear_probe_acc` | 用 R² 当信息量——多类 one-hot 上它会发散，**必换成有界的准确率** |
| `majority_baseline` | 拿 `1/n_class` 当底板——类别不平衡时它会骗人 |
| `validity` | 数值越界、发放出窗、**以及"没赢过多数类"** |
| `assert_patched` | 字符串替换静默失败 |

## 一条硬规矩

**任何读数先过 `validity`，再进结论。** 不过的读数记 `invalid`，不计入平均。

## 底板为什么不能是 1/n_class

类别平衡时二者相等；一旦不平衡，`1/n_class` 会低得离谱。历史上出现过一次
"acc 0.376 / 随机 0.125"，看着像 3 倍随机，其实**多数类本身就是 0.376** ——
那不是信息，是类别先验。底板必须**实测**。
"""

from __future__ import annotations

# ruff: noqa: N803, N806  —— 线性代数里 X/W/A/C 是惯例
import math

import torch

#: 工作点窗口：低于它网络几乎不动，高于它饱和
FIRING_WINDOW = (0.03, 0.20)
TARGET_FIRING = 0.08
#: 判定"赢过底板"用几个标准误。2σ 是统计惯例，不是为某批数据挑的常数。
N_SIGMA = 2.0


def scale_grid(lo: float = 0.4, hi: float = 12.0, ratio: float = 1.15):
    """**等比细网格**。

    发放率对输入尺度是断崖式的（实测 ring：scale 1.0 → 0.0027，2.0 → 0.311），
    粗网格会整段跨过可用窗口 —— 这一天因此废掉三个臂。
    """
    out, s = [], lo
    while s <= hi:
        out.append(round(s, 4))
        s *= ratio
    return out


def calibrate(
    run_at_scale,
    scales: list[float] | None = None,
    window: tuple[float, float] = FIRING_WINDOW,
    target: float = TARGET_FIRING,
):
    """返回 `(ok, best_scale, firing)`。**ok=False 时调用方必须放弃该臂。**"""
    best = None
    for sc in scales or scale_grid():
        try:
            fr = float(run_at_scale(sc))
        except Exception:
            continue
        if not math.isfinite(fr):
            continue
        if window[0] <= fr <= window[1]:
            score = abs(fr - target)
            if best is None or score < best[2]:
                best = (sc, fr, score)
    if best is None:
        return False, None, float("nan")
    return True, best[0], best[1]


#: 探针的正则网格。**λ 由训练集内部的一刀验证来选**，不由实验者拍。
LAM_GRID = (1e-2, 1e-1, 1.0, 10.0)


def _ridge_fit(Xtr, Ytr, n_class: int, lam_ratio: float):
    Yh = torch.nn.functional.one_hot(Ytr, n_class).float()
    lam = lam_ratio * Xtr.shape[0]
    A = Xtr.T @ Xtr + lam * torch.eye(Xtr.shape[1], device=Xtr.device)
    return torch.linalg.solve(A, Xtr.T @ Yh)


def select_lam_ratio(X, Y, cut: int, n_class: int, grid=LAM_GRID, hold: float = 0.2):
    """在**训练集内部**再切一刀选 λ。**这是仪器标定，不是调判据。**

    2026-11-16：d=4096、n=6144 时固定 λ=1e-2 会把训练集背下来
    （train 0.70 / test 0.15），而 λ=1 给出 test 0.24 —— 差 9pp，
    **比要测的效应还大**。λ 必须选，而且两臂必须用同一套选法。
    """
    inner = max(1, int(cut * (1.0 - hold)))
    mu, sd = X[:inner].mean(0), X[:inner].std(0) + 1e-6
    Xi, Xv = (X[:inner] - mu) / sd, (X[inner:cut] - mu) / sd
    best, best_acc = grid[0], -1.0
    for lr in grid:
        W = _ridge_fit(Xi, Y[:inner], n_class, lr)
        a = float((Xv @ W).argmax(1).eq(Y[inner:cut]).float().mean())
        if a > best_acc:
            best, best_acc = lr, a
    return best


def probe(X, Y, cut: int, n_class: int, lam_ratio: float | None = None):
    """返回 `(准确率, 用到的 lam_ratio)`。

    **不用 R²**：多类 one-hot 上 `1 − MSE/Var` 会大幅为负、且可发散
    （实测出现过 −5751），那时它不是"模型差"，是估计器坏了。
    准确率有界、可与底板直接比，是更稳的信息下界。
    """
    if lam_ratio is None:
        lam_ratio = select_lam_ratio(X, Y, cut, n_class)
    Xtr, Xte, Ytr, Yte = X[:cut], X[cut:], Y[:cut], Y[cut:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    W = _ridge_fit(Xtr, Ytr, n_class, lam_ratio)
    pred = (Xte @ W).argmax(1)
    return float((pred == Yte).float().mean()), lam_ratio


def linear_probe_acc(X, Y, cut: int, n_class: int, lam_ratio: float | None = None) -> float:
    """只要准确率时的薄壳。`lam_ratio=None` 表示按 `select_lam_ratio` 自动选。"""
    return probe(X, Y, cut, n_class, lam_ratio)[0]


def mlp_probe(
    X,
    Y,
    cut: int,
    n_class: int,
    hidden: int = 256,
    steps: int = 600,
    seeds=(0, 1, 2),
    batch: int = 8192,
):
    """非线性探针，返回 `(均值, 每次种子的值列表)`。

    超参与 `scripts/dim_ladder.py:mlp` **逐字相同** —— 这一条很重要：
    此前所有"环 ≫ 随机"的证据都是那个探针给的。若换成别的超参，
    新旧数字就不可比，等于偷偷换了一把尺子。

    **为什么会需要它**：本仓库已经用九轮实验确认"线性读出是瓶颈"
    （A2–A7、R3–R11）。用线性探针去判"状态里有没有信息"，
    测到的是那根瓶颈，不是信息本身。
    """
    Xtr, Xte, Ytr, Yte = X[:cut], X[cut:], Y[:cut], Y[cut:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    out = []
    for sd_seed in seeds:
        torch.manual_seed(sd_seed)
        net = torch.nn.Sequential(
            torch.nn.Linear(Xtr.shape[1], hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, n_class),
        ).to(Xtr.device)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        for _ in range(steps):
            perm = torch.randperm(Xtr.shape[0], device=Xtr.device)[:batch]
            opt.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(net(Xtr[perm]), Ytr[perm]).backward()
            opt.step()
        with torch.no_grad():
            out.append(float((net(Xte).argmax(-1) == Yte).float().mean()))
        del net
    return sum(out) / len(out), out


def majority_baseline(Ytr, Yte, n_class: int) -> float:
    """**实测底板**：在训练集上取最多的那一类，在测试集上量准确率。

    `1/n_class` 只在类别平衡时才是底板。历史上有一次 "acc 0.376 / 随机 0.125"
    看着像 3 倍随机，其实多数类本身就是 0.376 —— 那不是信息，是类别先验。
    """
    counts = torch.bincount(Ytr.reshape(-1), minlength=n_class)
    top = int(counts.argmax())
    return float((Yte.reshape(-1) == top).float().mean())


def probe_gain(acc: float, floor: float) -> float:
    """地板以上用掉了多少可用空间，∈ `[0, 1]`。**效果量，不是显著性。**"""
    return (acc - floor) / max(1e-9, 1.0 - floor)


def validity(
    acc: float,
    firing: float,
    floor: float,
    n_test: int,
    n_sigma: float = N_SIGMA,
) -> tuple[bool, list[str]]:
    """数值合法性闸。返回 `(ok, 原因列表)`。

    `floor` 必须是 `majority_baseline(...)` 的**实测值**，不是 `1/n_class`。
    """
    bad = []
    for name, v in (("acc", acc), ("firing", firing), ("floor", floor)):
        if not math.isfinite(v):
            bad.append(f"{name} 非有限值 {v}")
            return False, bad  # 后面的比较都没有意义
    if not (0.0 <= acc <= 1.0):
        bad.append(f"acc 越界 {acc}")
    if not (0.0 <= firing <= 1.0):
        bad.append(f"firing 越界 {firing}")
    if firing < FIRING_WINDOW[0] or firing > FIRING_WINDOW[1]:
        bad.append(f"发放 {firing:.4f} 出窗 {FIRING_WINDOW}")
    se = math.sqrt(max(acc * (1.0 - acc), 1e-12) / max(n_test, 1))
    if acc <= floor + n_sigma * se:
        bad.append(
            f"acc {acc:.4f} 未高过底板 {floor:.4f}+{n_sigma:.0f}σ({se:.4f})"
            "（在底板上，该读数不是信息）"
        )
    return (len(bad) == 0), bad


def assert_patched(text: str, old: str) -> None:
    """**字符串替换必须命中**。

    这一天至少三次"以为改了、其实 replace 什么也没做"，其中一次还让我
    基于未生效的修改去解读结果。
    """
    if old not in text:
        raise RuntimeError(f"补丁未命中，原文里找不到：{old[:60]!r}")


#: 任务生成分布 `pref` 的固定种子。**训练集与测试集必须共用同一个 `pref`**，
#: 否则两边不是同一个任务 —— 见 `seeded_dirichlet`。
PREF_SEED = 0


def seeded_dirichlet(alpha: float, k: int, seed: int = PREF_SEED) -> torch.Tensor:
    """**确定性**的 Dirichlet 抽样。

    `torch.distributions.Dirichlet(...).sample()` 走的是**全局** RNG，不吃局部
    `torch.Generator`。后果很隐蔽：同一个 seed 调两次会拿到两个不同的 `pref`，
    于是 `make_task(train)` 与 `make_task(test)` **是两个不同的任务**，
    测试集上的准确率必然掉到随机 —— 却完全不像 bug，像"这个结构没用"。

    2026-11-16 这个坑同时存在于 `manifold_big` / `bio_all` / `hyperbolic_readout`
    / `state_info` 四个脚本里，整天的"未定"结果全部由此报废。

    `fork_rng` 保证不改动调用方的全局 RNG 状态。
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return torch.distributions.Dirichlet(torch.full((k,), alpha)).sample((k,))
