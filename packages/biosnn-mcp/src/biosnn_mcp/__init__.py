"""BioSNN-Plug 阶段五 MCP：只读版（M0/M1 替身）+ 提议队列写版本（M2）。"""

from .consolidate import ConsolidationReport, MemoryItem, SleepConsolidator
from .context import ContextCode, ZScoreTopKEncoder
from .core import CognitiveCore, InMemoryCore, MemoryEntry, ToolError
from .dialogue import Scenario, Turn, WritableMemory, build_scenarios, run_session
from .embeddings import HashEmbedding, OllamaEmbedding
from .episodic import EpisodicStore
from .llm import LLMAdapter, MockLLM
from .llm_ollama import OllamaLLM
from .proposals import Gate, GateConfig, ProposalQueue
from .scheduling import SchedulingPolicy
from .server import (
    PROTOCOL_VERSION,
    TOOLS,
    WRITE_TOOLS,
    AuditLog,
    handle_message,
    main_write,
    serve_stdio,
    write_tools,
)
from .trace import SCHEMA_VERSION, DialogueTrace, TurnRecord
from .trigger import TriggerDecision, TriggerPolicy
from .trigger_spiking import SpikingTrigger
from .writecore import WriteCore

__all__ = [
    "PROTOCOL_VERSION",
    "SCHEMA_VERSION",
    "TOOLS",
    "WRITE_TOOLS",
    "AuditLog",
    "CognitiveCore",
    "ConsolidationReport",
    "ContextCode",
    "DialogueTrace",
    "EpisodicStore",
    "Gate",
    "GateConfig",
    "HashEmbedding",
    "InMemoryCore",
    "LLMAdapter",
    "MemoryEntry",
    "MemoryItem",
    "MockLLM",
    "OllamaEmbedding",
    "OllamaLLM",
    "ProposalQueue",
    "Scenario",
    "SchedulingPolicy",
    "SleepConsolidator",
    "SpikingTrigger",
    "ToolError",
    "TriggerDecision",
    "TriggerPolicy",
    "Turn",
    "TurnRecord",
    "WritableMemory",
    "WriteCore",
    "ZScoreTopKEncoder",
    "__version__",
    "build_scenarios",
    "handle_message",
    "main_write",
    "run_session",
    "serve_stdio",
    "write_tools",
]

from ._version import __version__  # 单一版本源
