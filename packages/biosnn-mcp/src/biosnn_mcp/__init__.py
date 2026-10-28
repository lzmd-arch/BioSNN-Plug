"""BioSNN-Plug 阶段五 M0：只读 MCP 工具骨架。"""

from .core import CognitiveCore, InMemoryCore, MemoryEntry, ToolError
from .server import PROTOCOL_VERSION, TOOLS, handle_message, serve_stdio

__all__ = [
    "PROTOCOL_VERSION",
    "TOOLS",
    "CognitiveCore",
    "InMemoryCore",
    "MemoryEntry",
    "ToolError",
    "handle_message",
    "serve_stdio",
]

__version__ = "0.1.0"
