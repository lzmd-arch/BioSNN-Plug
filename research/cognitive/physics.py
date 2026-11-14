"""Acrobot 的物理约束：把预测状态投影回**物理可行集**。

状态是 ``[cos θ1, sin θ1, cos θ2, sin θ2, ω1, ω2]``（gym ``Acrobot-v1``）。

与"裁到训练范围"的区别（那一条已被证否，见 `sweep_results/world_model_cubic/README.md`）：
这里的每一条都来自**物理**或**运动学**，与数据覆盖无关，因此不会把目标区域一起裁掉。
"""

from __future__ import annotations

import numpy as np

DT = 0.2
"""Acrobot-v1 的 ``dt``（gym 默认 0.2 s）。"""


def unit_circle(states: np.ndarray) -> np.ndarray:
    """C1：``(cos θ, sin θ)`` 必须落在单位圆上。"""
    out = np.array(states, dtype=np.float64, copy=True)
    for lo in (0, 2):
        pair = out[:, lo : lo + 2]
        out[:, lo : lo + 2] = pair / np.maximum(np.linalg.norm(pair, axis=1, keepdims=True), 1e-12)
    return out


def angles(states: np.ndarray) -> np.ndarray:
    s = np.atleast_2d(states)
    return np.stack([np.arctan2(s[:, 1], s[:, 0]), np.arctan2(s[:, 3], s[:, 2])], axis=1)


def kinematic(previous: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """C2：**角度块与速度块必须一致**。

    由 ``predicted`` 的角速度推角度：``θ = θ_prev + ω·dt``（解缠后取回最近的等价角）。
    现在的模型把这两块当独立变量预测，实测角度与速度可以互相矛盾——而物理上不可能。
    """
    prev = np.atleast_2d(previous)
    pred = np.array(predicted, dtype=np.float64, copy=True)
    theta_prev = angles(prev)
    omega = pred[:, 4:6]
    theta = theta_prev + omega * DT
    out = np.array(pred, copy=True)
    out[:, 0] = np.cos(theta[:, 0])
    out[:, 1] = np.sin(theta[:, 0])
    out[:, 2] = np.cos(theta[:, 1])
    out[:, 3] = np.sin(theta[:, 1])
    return out


#: Acrobot-v1 的物理常数与力矩上限。
M1 = M2 = 1.0
L1 = 1.0
LC1 = LC2 = 0.5
I1 = I2 = 1.0
G = 9.8
TAU_MAX = 1.0


def energy(states: np.ndarray) -> np.ndarray:
    """总机械能（势能 + 动能），用标准 Acrobot 参数。"""
    s = np.atleast_2d(states)
    theta1 = np.arctan2(s[:, 1], s[:, 0])
    theta2 = np.arctan2(s[:, 3], s[:, 2])
    w1, w2 = s[:, 4], s[:, 5]
    potential = -(M1 * LC1 + M2 * L1) * G * np.cos(theta1) - M2 * LC2 * G * np.cos(theta1 + theta2)
    kinetic = (
        0.5 * M1 * (LC1**2) * w1**2
        + 0.5 * M2 * ((L1**2) * w1**2 + (LC2**2) * w2**2 + 2 * L1 * LC2 * w1 * w2 * np.cos(theta2))
        + 0.5 * (I1 + I2) * w1**2
        + 0.5 * I2 * w2**2
    )
    return potential + kinetic


def energy_bound(previous: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """C3：单步能量变化不能超过 ``τ_max · |Δθ|``；超了就整体缩放速度把动能拉回来。"""
    prev = np.atleast_2d(previous)
    pred = np.array(predicted, dtype=np.float64, copy=True)
    delta_theta = np.abs(angles(pred) - angles(prev)).sum(axis=1)
    delta_energy = energy(pred) - energy(prev)
    budget = TAU_MAX * delta_theta + 1e-6
    excess = np.abs(delta_energy) > budget
    if np.any(excess):
        pe = energy(pred) - (
            0.5 * M1 * (LC1**2) * pred[:, 4] ** 2
            + 0.5
            * M2
            * (
                (L1**2) * pred[:, 4] ** 2
                + (LC2**2) * pred[:, 5] ** 2
                + 2 * L1 * LC2 * pred[:, 4] * pred[:, 5] * np.cos(angles(pred)[:, 1])
            )
            + 0.5 * (I1 + I2) * pred[:, 4] ** 2
            + 0.5 * I2 * pred[:, 5] ** 2
        )
        ke = energy(pred) - pe
        allow = np.clip(np.abs(energy(prev) + np.sign(delta_energy) * budget - pe), 1e-9, None)
        scale = np.sqrt(np.clip(allow / np.maximum(ke, 1e-12), 0.0, 1.0))
        scale = np.where(excess, scale, 1.0)
        pred[:, 4] *= scale
        pred[:, 5] *= scale
    return pred


def apply_all(previous: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """C1 → C4 → C2 → C3。

    顺序有讲究：先回单位圆（不变量），再用**力矩上限**压住速度跳变（硬物理），
    然后让角度服从速度（运动学），最后用能量预算收口。
    """
    out = torque_bound(previous, unit_circle(predicted))
    return energy_bound(previous, kinematic(previous, out))


def torque_bound(previous: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """C4：**力矩上限给出的角加速度上限**。

    单步角速度变化不能超过 ``τ_max / I · dt``。这是硬物理：力矩最大也就那么大，
    速度不可能一步跳变。它直接针对实测的溢出维（``ω1/ω2``），而且与数据覆盖无关。
    """
    prev = np.atleast_2d(previous)
    pred = np.array(predicted, dtype=np.float64, copy=True)
    max_delta = TAU_MAX / min(I1, I2) * DT
    for dim in (4, 5):
        delta = np.clip(pred[:, dim] - prev[:, dim], -max_delta, max_delta)
        pred[:, dim] = prev[:, dim] + delta
    return pred


def apply_adaptive(
    previous: np.ndarray,
    predicted: np.ndarray,
    envelope: tuple[np.ndarray, np.ndarray],
    *,
    tolerance: float = 1.0,
) -> np.ndarray:
    """**自适应**：C1 与 C4 每步都上（它们是无条件成立的硬约束），
    C2/C3 只在预测**跑出数据包络**时才上。

    为什么：C2 是覆盖式的，短前瞻时模型的角度本来就准，覆盖反而引入误差；
    长前瞻时它才救命。判据用"是否出了包络"，不用前瞻步数——后者是拍脑袋的。
    """
    # **不套 C4**：实测它把长前瞻的改善整个抹掉（见 README 第二轮）。
    # 连续时间的力矩界要按积分器重新标定，不能直接裁单步 delta。
    out = unit_circle(predicted)
    low, high = envelope
    span = np.maximum(high - low, 1e-9)
    outside = np.any((out < low - tolerance * span) | (out > high + tolerance * span), axis=1)
    if np.any(outside):
        constrained = energy_bound(previous, kinematic(previous, out))
        out = np.where(outside[:, None], constrained, out)
    return out
