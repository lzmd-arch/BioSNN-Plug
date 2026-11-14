"""persona-core：**通用**的人格与记忆核（不绑定任何宿主程序）。"""

from .init_memory import BUILTIN_REF, SCHEMA, init_memory, manifest

__all__ = ["BUILTIN_REF", "SCHEMA", "init_memory", "manifest"]

from .influence import Influence
from .state import SelfState

__all__ += ["Influence", "SelfState"]
