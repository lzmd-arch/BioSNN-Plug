"""成功偏移 / σR 的测量（计划书 §3.2、§七 第一阶段的验收指标）。

**⚠️ 破坏性变更免责期：至 2027-09**（计划书 §12.5、ADR-0005）。

计划书 §七 第一阶段的验收标准里写着「**成功偏移 < 10%σR**」。这个指标不是可选的——
它量的正是 R-STDP 那个结构性失败模式：

> 成功偏移达到成功信号标准差的 ~25%（σR）时，R-STDP 就无法学习目标任务。更严重的是，
> 当偏移 < −0.4σR（平均成功信号为负）时，学习后的性能甚至低于学习前——R-STDP 不仅无法
> 学习，还会导致遗忘已学会的技能。

出处 `docs/references.md` [3] 记的原文措辞是 **"±25% of the SD (σR)"（双向）**。

## 定义（写死在这里，不留事后解释的余地）

    S       = 成功信号序列（每个时间步一个标量）
    偏移     = mean(S)                    ← 有符号；负号意味着"平均成功信号为负"那条退化路径
    σR      = std(S)，**总体标准差（ddof = 0）**
    比值     = |偏移| / σR

判据：**比值 < 0.10**（计划书 §七）。

用总体标准差而非样本标准差，是为了让这个数在"只跑了几十个回合"和"跑了几千个回合"时
含义一致——样本标准差在小样本下会系统性偏大，把比值压小，等于偷偷放宽判据。

**σR 的语义要说清**：原文的 σR 是成功信号 R 的标准差。本项目的"成功信号"是
``S = R − V(s)``（见 :mod:`research.rstdp.rstdp`），所以 σR 是**这个序列**的标准差，
不是原始奖励的标准差。两者不同，报告里必须写明用的是哪一个。

## 为什么要同时报"全程"与"末段"

全程的偏移会把训练早期的混乱算进去；末段的偏移反映的是**收敛后**的偏差，那才是
"能不能继续学"的关键。两个都报，并**以末段为准**判验收。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = ["BiasReport", "BiasTracker"]

#: 计划书 §七 的验收阈值。
OFFSET_THRESHOLD = 0.10


@dataclass
class BiasReport:
    """一组成功信号的偏移统计。"""

    n: int
    offset: float
    sigma: float

    @property
    def ratio(self) -> float:
        """``|偏移| / σR``。σR 为 0 时返回 ``inf``（除非偏移也是 0，那时是 0）。

        这一条必须显式处理：一个恒定的成功信号既不是"无偏"也不是"有偏"——它意味着
        Critic 与奖励完全同步（或奖励恒定），此时比值无定义。返回 ``inf`` 会让它**失败**
        而不是蒙混过关，这是刻意的：无定义不等于达标。
        """
        if self.sigma <= 0:
            return 0.0 if abs(self.offset) <= 0 else float("inf")
        return abs(self.offset) / self.sigma

    @property
    def passed(self) -> bool:
        return self.ratio < OFFSET_THRESHOLD

    def render(self) -> str:
        verdict = "达到" if self.passed else "未达到"
        return (
            f"n={self.n}  偏移={self.offset:+.5f}  σR={self.sigma:.5f}  "
            f"|偏移|/σR={self.ratio:.4f}（阈值 < {OFFSET_THRESHOLD}）→ {verdict}"
        )


@dataclass
class BiasTracker:
    """累积成功信号，产出 :class:`BiasReport`。

    Attributes:
        tail_window: 末段统计的窗口长度（时间步数）。``None`` 表示不报末段。
    """

    tail_window: int | None = 2000
    _values: list[float] = field(default_factory=list, init=False)

    def record(self, success_signal: float) -> None:
        self._values.append(float(success_signal))

    def extend(self, values) -> None:
        self._values.extend(float(v) for v in values)

    def __len__(self) -> int:
        return len(self._values)

    def _report(self, values: list[float]) -> BiasReport:
        if not values:
            return BiasReport(n=0, offset=0.0, sigma=0.0)
        tensor = torch.tensor(values, dtype=torch.float64)
        return BiasReport(
            n=len(values),
            offset=float(tensor.mean().item()),
            sigma=float(tensor.std(unbiased=False).item()),
        )

    @property
    def overall(self) -> BiasReport:
        """全程统计。"""
        return self._report(self._values)

    @property
    def tail(self) -> BiasReport:
        """末段统计；``tail_window`` 为 ``None`` 或数据不足时退化为全程。"""
        if self.tail_window is None or len(self._values) <= self.tail_window:
            return self.overall
        return self._report(self._values[-self.tail_window :])

    def render(self) -> str:
        return f"全程：{self.overall.render()}\n末段：{self.tail.render()}"
