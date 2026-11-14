"""核心 → 云端的**唯一通道**：一个可消融、可测量、有界的包。

云端只吃文本，所以"影响云端"没有第二条路——要么写进提示词，要么影响不到。
这里把要写进去的东西收成一个对象，并且**必须能一键消融**（:meth:`ablated`）：
没有消融就分不清"是她影响了回答"还是"回答本来就这样"。

三段，各自可零：

| 段 | 内容 | 消融后 |
| :--- | :--- | :--- |
| `self_state` | 她此刻的状态（:class:`~persona_core.state.SelfState`） | 只留慢变量 |
| `memories` | 检索到的原文片段，**逐条标明来自记忆** | 清空 |
| `directives` | 由状态与伦理层推出来的行为指令（≤3 条） | 清空 |
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .state import SelfState


@dataclass(frozen=True)
class Influence:
    self_state: SelfState
    memories: tuple[str, ...] = ()
    directives: tuple[str, ...] = ()
    extra: dict = field(default_factory=dict)

    MAX_MEMORIES = 5
    MAX_DIRECTIVES = 3
    MAX_CHARS = 900

    def render_prompt_block(self) -> str:
        """给云端的那一段。**有硬上限**——云端每轮都要吃，拼长了就是成本与漂移。"""
        parts = ["【她的状态】" + self.self_state.render()]
        if self.memories:
            lines = [f"- {m}" for m in self.memories[: self.MAX_MEMORIES]]
            parts.append("【从记忆里查到的】（原文，不是你的推测）\n" + "\n".join(lines))
        if self.directives:
            lines = [f"- {d}" for d in self.directives[: self.MAX_DIRECTIVES]]
            parts.append("【这一轮的行为约束】\n" + "\n".join(lines))
        text = "\n\n".join(parts)
        return text if len(text) <= self.MAX_CHARS else text[: self.MAX_CHARS]

    def to_state_vector(self) -> list[float]:
        return self.self_state.to_vector()

    def ablated(self) -> Influence:
        """**消融版**：清记忆、清指令、快变量归零。用来量"她到底影响了多少"。"""
        return replace(self, self_state=self.self_state.zeroed(), memories=(), directives=())

    def is_empty(self) -> bool:
        a = self.ablated()
        return (
            not a.memories
            and not a.directives
            and a.render_prompt_block() == self.render_prompt_block()
        )
