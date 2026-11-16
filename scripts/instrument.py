"""仪器：**先把"能不能测"这件事做成可复用的函数**，再谈结论。

## 为什么要有这个文件

2026-11-16 一天之内，同一类工程错误让约 12 轮实验作废，其中两次还跑出了
"看起来通过"的假结果（R² = −1.6×10⁹ 被判过、准确率 0.000 被当结论）。

四道闸，全部集中在这里，**任何实验都必须先过它们**：

| 闸 | 拦什么 |
| :--- | :--- |
| `calibrate` | 网络静默 / 饱和；网格跨过窄窗口 |
| `linear_probe_acc` | 用 R² 当信息量——多类 one-hot 上它会发散，**必换成有界的准确率** |
| `validity` | 数值越界（NaN/Inf/负准确率/发放出窗） |
| `assert_patched` | 字符串替换静默失败 |

## 一条硬规矩

**任何读数先过 `validity`，再进结论。** 不过的读数记 `invalid`，不计入平均。
"""

from __future__ import annotations

# ruff: noqa: N803, N806  —— 线性代数里 X/W/A/C 是惯例
import math

import torch

#: 工作点窗口：低于它网络几乎不动，高于它饱和
FIRING_WINDOW = (0.03, 0.20)
TARGET_FIRING = 0.08


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


def linear_probe_acc(X, Y, cut: int, n_class: int, lam_ratio: float = 1e-2) -> float:
    """线性探针的**准确率**（有界 [0,1]）。

    **不用 R²**：多类 one-hot 上 `1 − MSE/Var` 会大幅为负、且可发散
    （实测出现过 −5751），那时它不是"模型差"，是估计器坏了。
    准确率有界、可与 `1/n_class` 直接比，是更稳的信息下界。
    """
    Xtr, Xte, Ytr, Yte = X[:cut], X[cut:], Y[:cut], Y[cut:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Yh = torch.nn.functional.one_hot(Ytr, n_class).float()
    lam = lam_ratio * Xtr.shape[0]
    A = Xtr.T @ Xtr + lam * torch.eye(Xtr.shape[1], device=X.device)
    W = torch.linalg.solve(A, Xtr.T @ Yh)
    pred = (Xte @ W).argmax(1)
    return float((pred == Yte).float().mean())


def validity(
    acc: float, firing: float, chance: float, min_gain: float = 0.15
) -> tuple[bool, list[str]]:
    """数值合法性闸。返回 `(ok, 原因列表)`。"""
    bad = []
    for name, v in (("acc", acc), ("firing", firing), ("chance", chance)):
        if not math.isfinite(v):
            bad.append(f"{name} 非有限值 {v}")
    if not (0.0 <= acc <= 1.0):
        bad.append(f"acc 越界 {acc}")
    if not (0.0 <= firing <= 1.0):
        bad.append(f"firing 越界 {firing}")
    if firing < FIRING_WINDOW[0] or firing > FIRING_WINDOW[1]:
        bad.append(f"发放 {firing:.4f} 出窗 {FIRING_WINDOW}")
    if acc < chance + min_gain:
        bad.append(f"acc {acc:.3f} < 随机 {chance:.3f}+{min_gain}（在底板上，不可比）")
    return (len(bad) == 0), bad


def assert_patched(text: str, old: str) -> None:
    """**字符串替换必须命中**。

    这一天至少三次"以为改了、其实 replace 什么也没做"，其中一次还让我
    基于未生效的修改去解读结果。
    """
    if old not in text:
        raise RuntimeError(f"补丁未命中，原文里找不到：{old[:60]!r}")
