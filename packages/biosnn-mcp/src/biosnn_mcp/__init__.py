"""BioSNN-Plug 阶段五 M0：只读 MCP 工具骨架。"""

from .core import CognitiveCore, InMemoryCore, MemoryEntry, ToolError
from .llm import LLMAdapter, MockLLM
from .server import PROTOCOL_VERSION, TOOLS, handle_message, serve_stdio
from .trace import SCHEMA_VERSION, DialogueTrace, TurnRecord

__all__ = [
    "PROTOCOL_VERSION",
    "SCHEMA_VERSION",
    "TOOLS",
    "CognitiveCore",
    "DialogueTrace",
    "InMemoryCore",
    "LLMAdapter",
    "MemoryEntry",
    "MockLLM",
    "ToolError",
    "TurnRecord",
    "handle_message",
    "serve_stdio",
]

__version__ = "0.1.0"
