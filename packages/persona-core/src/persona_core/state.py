"""**她自己的状态**：让核心知道自己此刻是什么、在哪、记得多少。

第三条要求是"她本身能够知道模型现在的状态"。这不是装饰——没有它，
"只说你真的做过的事"那条边界就只能靠猜：她不知道这轮跑的是哪个读者模型、
不知道自己是离线还是在线、不知道自己记得多少条。

这里只做两件事：把状态**收成一个不可变对象**，再把它**渲染成两样东西**——
给云端看的文字块，和给融合模型看的数值向量。**同一个对象，两种出口。**
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SelfState:
    """她此刻的状态。全部是**能查到的事实**，没有一项是估计。"""

    persona_name: str = "千岁（阿螈）"
    birth_date: str = "2026-11-16"
    age_at_birth: int = 20

    persona_version: int = 0
    persona_in_weights: bool = False  # 人格是否已进权重（而不是只靠锚文本）
    memory_count: int = 0
    session_turns: int = 0
    session_seconds: float = 0.0

    #: 本轮实际在跑哪个读者模型。**这是"换模型人格不变"的可观测证据**。
    backbone: str = ""
    online: bool = True  # 云端可达 = True
    retrieval_used: bool = False  # 这一轮有没有真的检索

    detail: dict = field(default_factory=dict)

    # -- 年龄 ---------------------------------------------------------------

    def age_years(self, now: float | None = None) -> float:
        import datetime as _dt

        birth = _dt.datetime.fromisoformat(self.birth_date + "T00:00:00").timestamp()
        at = time.time() if now is None else float(now)
        return self.age_at_birth + max(0.0, (at - birth) / (365.2425 * 86400.0))

    # -- 出口一：给云端的文字 ------------------------------------------------

    def render(self, now: float | None = None) -> str:
        """**短**。云端每轮都要吃它，所以只写会改变回答方式的事实。"""
        bits = [
            f"你的名字是{self.persona_name}，出生于 {self.birth_date}，"
            f"现在 {int(self.age_years(now))} 岁。",
            f"这一轮由「{self.backbone or '未知读者模型'}」来读你；"
            f"记忆库里现在有 {self.memory_count} 条。",
            "云端可用。" if self.online else "现在是离线状态，回答能力受限，要说清楚。",
        ]
        if self.persona_in_weights:
            bits.append("你的身份与行为约束已经长在权重里，不依赖这轮有没有带上人设文本。")
        if self.session_turns:
            bits.append(f"这次对话已经进行了 {self.session_turns} 轮。")
        if self.retrieval_used:
            bits.append("这一轮你查过记忆；引用记忆时说明是从记忆里来的。")
        return "".join(bits)

    # -- 出口二：给融合模型的数值 --------------------------------------------

    N_DIM = 8

    def to_vector(self, now: float | None = None) -> list[float]:
        """固定 8 维，供 FiLM 调制器用。**归一化到 [0,1]**，别让量纲失衡。

        前 4 维是慢变量（身份/年龄），后 4 维是快变量（这一轮的状态）。
        """
        import math

        return [
            min(1.0, self.age_years(now) / 100.0),  # 年龄（慢）
            min(1.0, self.persona_version / 20.0),  # 人格版次（慢）
            1.0 if self.persona_in_weights else 0.0,  # 人格是否进权重
            min(1.0, math.log1p(self.memory_count) / 12.0),  # 记忆规模（慢）
            min(1.0, self.session_turns / 50.0),  # 会话轮次（快）
            min(1.0, self.session_seconds / 3600.0),  # 会话时长（快）
            1.0 if self.online else 0.0,  # 在线（快）
            1.0 if self.retrieval_used else 0.0,  # 这轮检索过（快）
        ]

    def zeroed(self) -> SelfState:
        """消融用：把**快变量**清零，慢变量保留。"""
        return SelfState(
            persona_name=self.persona_name,
            birth_date=self.birth_date,
            age_at_birth=self.age_at_birth,
            persona_version=self.persona_version,
            persona_in_weights=self.persona_in_weights,
            memory_count=self.memory_count,
        )
