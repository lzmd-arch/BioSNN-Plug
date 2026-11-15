"""统一的 chat template 渲染。

**为什么需要这一层**：Qwen3 的模板默认 `enable_thinking=True`，会往提示里塞
` thinking` 前缀，训练时模型会连"思考段"一起学，评测时又会先生成一堆推理再给答案——
换底座那一步（④）如果不处理这个，测的就不是人格，是思考模式。

这里统一传 `enable_thinking=False`；对没有这个变量的模板（Qwen2.5）是无害的。
"""

from __future__ import annotations


def render(tok, messages: list[dict], add_generation_prompt: bool = True) -> str:
    kw = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": False,
    }
    try:
        return tok.apply_chat_template(messages, **kw)
    except TypeError:
        kw.pop("enable_thinking")
        return tok.apply_chat_template(messages, **kw)
