"""唤醒通道：一个标量状态，调制其它组件的增益（ADR-0016）。

**它是什么**：动力学轴里那个一直缺失的组件——此前"唤醒"只以形容词出现在计划书里，
没有状态变量、没有接口、没有验收。本模块给它三样东西：一个标量、一条更新规则、一个作用点。

**它不是什么**：
* 不是学习规则——**不参与任何梯度**，也不由反向传播训练；
* 不是生物构念的等价物——它是本项目自选的控制器变量（ADR-0016 写明了这一条）；
* 不直接改权重——它只调制**其它组件的增益**（巩固频率、探索温度、唤醒阈值）。

更新规则是**指数滑动平均**：目标由输入强度与近期发放率线性合成，状态朝目标走 `tau` 那一份。
没有随机性，因此同一条输入序列必然给出同一条轨迹——这是可复现的前提。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ArousalChannel", "ArousalConfig"]


@dataclass(frozen=True)
class ArousalConfig:
    """唤醒通道的常数。全部**不训练**。

    Attributes:
        baseline: 初始与静息值。
        input_gain: 输入强度对目标的权重。
        firing_gain: 近期发放率对目标的权重。
        tau: 每次观测朝目标移动的比例，``0 < tau <= 1``。
        lo: 状态下界。
        hi: 状态上界。
        gain_span: ``gain()`` 在静息点两侧的调制幅度（``0`` 表示不调制）。
    """

    baseline: float = 0.3
    input_gain: float = 0.6
    firing_gain: float = 0.8
    tau: float = 0.1
    lo: float = 0.0
    hi: float = 1.0
    gain_span: float = 0.5

    def __post_init__(self) -> None:
        if not self.lo < self.hi:
            raise ValueError(f"要求 lo < hi，收到 {self.lo} 与 {self.hi}。")
        if not self.lo <= self.baseline <= self.hi:
            raise ValueError(f"baseline 必须落在 [{self.lo}, {self.hi}]，收到 {self.baseline}。")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError(f"tau 必须落在 (0, 1]，收到 {self.tau}。")
        if self.input_gain < 0 or self.firing_gain < 0:
            raise ValueError("input_gain / firing_gain 不能为负。")
        if not 0.0 <= self.gain_span <= 1.0:
            raise ValueError(f"gain_span 必须落在 [0, 1]，收到 {self.gain_span}。")


class ArousalChannel:
    """一个标量状态 + 一条朝目标收缩的更新规则。**不可微、不训练、无随机性。**"""

    def __init__(self, config: ArousalConfig | None = None) -> None:
        self.config = config or ArousalConfig()
        self._arousal = self.config.baseline

    @property
    def arousal(self) -> float:
        """当前唤醒值，位于 ``[lo, hi]``。"""
        return self._arousal

    def target(self, *, input_strength: float, firing_rate: float) -> float:
        """当前的"想去的地方"：两路证据的线性合成，夹在界内。"""
        raw = self.config.input_gain * float(input_strength) + self.config.firing_gain * float(
            firing_rate
        )
        return min(max(raw, self.config.lo), self.config.hi)

    def observe(self, *, input_strength: float, firing_rate: float) -> float:
        """观测一次（输入强度 + 近期发放率），返回更新后的唤醒值。"""
        aim = self.target(input_strength=input_strength, firing_rate=firing_rate)
        moved = self._arousal + self.config.tau * (aim - self._arousal)
        self._arousal = min(max(moved, self.config.lo), self.config.hi)
        return self._arousal

    def gain(self, base: float) -> float:
        """把一个基础量按当前唤醒调制后返回。

        静息点 ``baseline`` 处返回原值；唤醒越高返回越大，越低返回越小，
        两侧幅度由 ``gain_span`` 限制在 ``[1 - span, 1 + span]``。
        """
        span = self.config.gain_span
        if span == 0.0:
            return float(base)
        width = self.config.hi - self.config.lo
        if width <= 0:
            return float(base)
        centred = (self._arousal - self.config.baseline) / width
        scale = 1.0 + span * 2.0 * centred
        return float(base) * min(max(scale, 1.0 - span), 1.0 + span)

    def reset(self) -> None:
        """回到静息值。用于每个回合/每次运行的开头，保证可复现。"""
        self._arousal = self.config.baseline

    def __repr__(self) -> str:
        return f"ArousalChannel(arousal={self._arousal:.3f}, baseline={self.config.baseline:.3f})"
