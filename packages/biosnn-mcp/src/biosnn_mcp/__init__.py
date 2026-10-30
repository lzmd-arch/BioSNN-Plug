"""BioSNN-Plug 阶段五 MCP：只读骨架（M0）+ 显式可写工具（M1）。"""

from .consolidate import ConsolidationReport, MemoryItem, SleepConsolidator
from .context import ContextCode, ZScoreTopKEncoder
from .core import CognitiveCore, InMemoryCore, MemoryEntry, ToolError
from .dialogue import Scenario, Turn, WritableMemory, build_scenarios, run_session
from .embeddings import HashEmbedding, OllamaEmbedding
from .episodic import EpisodicStore
from .llm import LLMAdapter, MockLLM
from .llm_ollama import OllamaLLM
from .scheduling import SchedulingPolicy
from .server import PROTOCOL_VERSION, TOOLS, AuditLog, handle_message, serve_stdio
from .trace import SCHEMA_VERSION, DialogueTrace, TurnRecord
from .trigger import TriggerDecision, TriggerPolicy
from .trigger_spiking import SpikingTrigger

__all__ = [
    "PROTOCOL_VERSION",
    "SCHEMA_VERSION",
    "TOOLS",
    "AuditLog",
    "CognitiveCore",
    "ConsolidationReport",
    "ContextCode",
    "DialogueTrace",
    "EpisodicStore",
    "HashEmbedding",
    "InMemoryCore",
    "LLMAdapter",
    "MemoryEntry",
    "MemoryItem",
    "MockLLM",
    "OllamaEmbedding",
    "OllamaLLM",
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
    "ZScoreTopKEncoder",
    "build_scenarios",
    "handle_message",
    "run_session",
    "serve_stdio",
]

__version__ = "0.2.0"
