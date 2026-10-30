"""MCP Server：JSON-RPC 2.0 over stdio，无第三方依赖。

支持的方法：``initialize``、``notifications/initialized``、``ping``、``tools/list``、
``tools/call``。

工具面分两层：

* **只读**（任何模式都可用）：``state.query``、``memory.search``、``trigger.status``、
  ``trigger.poll``；
* **提议**（服务器必须以 ``--allow-write`` 启动）：``propose.memory.write``、
  ``propose.memory.consolidate``、``propose.trigger.arm``、``propose.trigger.disarm``、
  ``propose.trigger.feed``。

**为什么叫 ``propose.*`` 而不是 ``write.*``**：这些工具**永远不会写真实记忆库**——它们只改
本服务器内存里的替身核。叫 ``write`` 会让名字与行为不符；叫 ``propose`` 则把契约写在名字上：
"这是一条提议，由替身核处置"。将来若要让提议产生真实记忆效果，接法不是把这些工具接线，
而是新增**动力学独占写**的 ``propose`` 队列 + 巩固门控消费。

默认只读是刻意的（ADR-0020 的 M0 边界）；``--audit-log`` 把每一次 ``tools/call`` 记成一行
JSONL（时间、工具、参数、成功与否），使调用可审计、可回放。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from ._version import __version__
from .core import CognitiveCore, InMemoryCore, ToolError
from .writecore import WriteCore

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "biosnn-mcp", "version": __version__}

_EMPTY_SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "state.query",
        "description": "Read the phase-5 cognitive-core status (read-only).",
        "inputSchema": _EMPTY_SCHEMA,
    },
    {
        "name": "memory.search",
        "description": "Search the in-memory index by token overlap (read-only).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "propose.memory.write",
        "description": "Propose a memory entry to the server's in-memory stand-in core (never a real memory store). Requires --allow-write.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "minLength": 1},
                "id": {"type": "string", "minLength": 1},
                "tags": {"type": "array", "items": {"type": "string", "minLength": 1}},
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    },
    {
        "name": "propose.memory.consolidate",
        "description": "Propose one sleep-consolidation pass over the stand-in core. Requires --allow-write.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "threshold": {"type": "number", "minimum": -1, "maximum": 1},
                "rounds": {"type": "integer", "minimum": 1, "maximum": 10},
                "noise_sigma": {"type": "number", "minimum": 0, "maximum": 1},
                "seed": {"type": "integer"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "trigger.status",
        "description": "Read whether the spiking trigger is armed (read-only).",
        "inputSchema": _EMPTY_SCHEMA,
    },
    {
        "name": "propose.trigger.arm",
        "description": "Propose arming the stand-in spiking trigger; a reason string is required. Requires --allow-write.",
        "inputSchema": {
            "type": "object",
            "properties": {"reason": {"type": "string", "minLength": 1}},
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "propose.trigger.disarm",
        "description": "Propose disarming the stand-in trigger and dropping pending events. Requires --allow-write.",
        "inputSchema": _EMPTY_SCHEMA,
    },
    {
        "name": "propose.trigger.feed",
        "description": "Propose one step of the stand-in spiking trigger with a salience in [0, 1]. Requires --allow-write.",
        "inputSchema": {
            "type": "object",
            "properties": {"salience": {"type": "number", "minimum": 0, "maximum": 1}},
            "required": ["salience"],
            "additionalProperties": False,
        },
    },
    {
        "name": "trigger.poll",
        "description": "Drain queued trigger events (read-only).",
        "inputSchema": _EMPTY_SCHEMA,
    },
]


#: 写版本（``biosnn-mcp-write``）**新增**的四个工具；只读版的工具表一个字都不改（W6）。
WRITE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "queue.status",
        "description": "Read proposal-queue depth, quota headroom and the gate configuration (read-only).",
        "inputSchema": _EMPTY_SCHEMA,
    },
    {
        "name": "queue.list",
        "description": "List proposals with their verdicts and reasons (read-only).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["pending", "accepted", "rejected", "reverted"],
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "memory.revert",
        "description": "Undo one accepted write by restoring its pre-write snapshot; the revert is audited. Requires a reason.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "seq": {"type": "integer", "minimum": 1},
                "reason": {"type": "string", "minLength": 1},
            },
            "required": ["seq", "reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "gate.configure",
        "description": "Adjust gate thresholds or quotas. A reason is required and enters the audit log.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "minLength": 1},
                "changes": {
                    "type": "object",
                    "properties": {
                        "duplicate_cosine": {"type": "number", "minimum": -1, "maximum": 1},
                        "accept_salience": {"type": "number", "minimum": 0, "maximum": 1},
                        "floor_salience": {"type": "number", "minimum": 0, "maximum": 1},
                        "quota_hourly": {"type": "integer", "minimum": 0},
                        "quota_daily": {"type": "integer", "minimum": 0},
                    },
                    "additionalProperties": False,
                },
            },
            "required": ["reason", "changes"],
            "additionalProperties": False,
        },
    },
]


def write_tools() -> list[dict[str, Any]]:
    """写版本的工具表：只读版的 9 个原样保留，两个 propose 工具多一个 ``proposal_id``。"""
    tools = copy.deepcopy(TOOLS)
    for tool in tools:
        if tool["name"] == "propose.memory.write":
            tool["description"] = (
                "Propose a memory entry to the queue; the consolidation gate decides. "
                "Requires a client-generated proposal_id (idempotency key)."
            )
            tool["inputSchema"]["properties"]["proposal_id"] = {
                "type": "string",
                "minLength": 1,
            }
            tool["inputSchema"]["properties"]["source"] = {
                "type": "string",
                "enum": ["user", "client", "llm"],
            }
            tool["inputSchema"]["required"] = ["text", "proposal_id"]
        elif tool["name"] == "propose.memory.consolidate":
            tool["inputSchema"]["properties"]["proposal_id"] = {
                "type": "string",
                "minLength": 1,
            }
            tool["inputSchema"]["required"] = ["proposal_id"]
    return tools + copy.deepcopy(WRITE_TOOLS)


@dataclass
class AuditLog:
    """工具调用审计：每次 ``tools/call`` 一行。``path=None`` 时只留在内存里（测试用）。"""

    path: Path | None = None
    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        *,
        tool: str,
        arguments: dict[str, Any],
        ok: bool,
        error: str | None = None,
    ) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tool": tool,
            "ok": ok,
            "error": error,
            "arguments": arguments,
        }
        self.entries.append(entry)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}


def _result(message_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _text_result(payload: dict[str, Any], *, is_error: bool = False) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
        "isError": is_error,
    }


def _reject_unknown(arguments: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(arguments) - allowed
    if unknown:
        raise ToolError(f"收到未知参数：{sorted(unknown)}")


def _is_write_core(core: CognitiveCore) -> bool:
    """写版本的核带队列；只读版的核没有这些方法。"""
    return hasattr(core, "queue_status")


def _call_tool(core: CognitiveCore, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name in ("queue.status", "queue.list", "memory.revert", "gate.configure"):
        if not _is_write_core(core):
            raise KeyError(name)
        if name == "queue.status":
            if arguments:
                raise ToolError("queue.status 不接受参数。")
            return _text_result(core.queue_status())
        if name == "queue.list":
            _reject_unknown(arguments, {"status", "limit"})
            return _text_result(
                core.queue_list(
                    status=arguments.get("status"),
                    limit=int(arguments.get("limit", 20)),
                )
            )
        if name == "memory.revert":
            _reject_unknown(arguments, {"seq", "reason"})
            if "seq" not in arguments or "reason" not in arguments:
                raise ToolError("memory.revert 需要 seq 与 reason。")
            return _text_result(core.revert(arguments.get("seq"), arguments.get("reason")))
        _reject_unknown(arguments, {"reason", "changes"})
        if "reason" not in arguments or "changes" not in arguments:
            raise ToolError("gate.configure 需要 reason 与 changes。")
        return _text_result(core.configure_gate(arguments.get("reason"), arguments.get("changes")))
    if name == "state.query":
        if arguments:
            raise ToolError("state.query 不接受参数。")
        return _text_result(core.state())
    if name == "trigger.status":
        if arguments:
            raise ToolError("trigger.status 不接受参数。")
        return _text_result(core.trigger_status())
    if name == "trigger.poll":
        if arguments:
            raise ToolError("trigger.poll 不接受参数。")
        return _text_result(core.poll())
    if name == "memory.search":
        _reject_unknown(arguments, {"query", "top_k"})
        hits = core.search(arguments.get("query"), arguments.get("top_k", 5))
        return _text_result({"hits": hits, "count": len(hits)})
    if name == "propose.memory.write":
        _reject_unknown(arguments, {"text", "id", "tags", "proposal_id", "source"})
        tags = arguments.get("tags") or []
        if not isinstance(tags, list):
            raise ToolError("tags 必须是数组。")
        if _is_write_core(core):
            if "proposal_id" not in arguments:
                raise ToolError("写版本需要 proposal_id（客户端生成的幂等键）。")
            return _text_result(
                core.propose_write(
                    arguments.get("text"),
                    arguments.get("id"),
                    tuple(tags),
                    proposal_id=arguments.get("proposal_id"),
                    source=str(arguments.get("source", "client")),
                )
            )
        payload = core.write(
            arguments.get("text"),
            arguments.get("id"),
            tuple(tags),
        )
        return _text_result(payload)
    if name == "propose.memory.consolidate":
        _reject_unknown(arguments, {"threshold", "rounds", "noise_sigma", "seed", "proposal_id"})
        if _is_write_core(core):
            if "proposal_id" not in arguments:
                raise ToolError("写版本需要 proposal_id（客户端生成的幂等键）。")
            return _text_result(
                core.propose_consolidate(
                    proposal_id=arguments.get("proposal_id"),
                    threshold=float(arguments.get("threshold", 0.95)),
                    rounds=int(arguments.get("rounds", 3)),
                    noise_sigma=float(arguments.get("noise_sigma", 0.05)),
                    seed=int(arguments.get("seed", 0)),
                )
            )
        payload = core.consolidate(
            threshold=float(arguments.get("threshold", 0.95)),
            rounds=int(arguments.get("rounds", 3)),
            noise_sigma=float(arguments.get("noise_sigma", 0.05)),
            seed=int(arguments.get("seed", 0)),
        )
        return _text_result(payload)
    if name == "propose.trigger.arm":
        _reject_unknown(arguments, {"reason"})
        return _text_result(core.arm(arguments.get("reason")))
    if name == "propose.trigger.disarm":
        if arguments:
            raise ToolError("trigger.disarm 不接受参数。")
        return _text_result(core.disarm())
    if name == "propose.trigger.feed":
        _reject_unknown(arguments, {"salience"})
        if "salience" not in arguments:
            raise ToolError("trigger.feed 需要 salience。")
        return _text_result(core.feed(arguments.get("salience")))
    raise KeyError(name)


def handle_message(
    core: CognitiveCore,
    message: dict[str, Any],
    *,
    audit: AuditLog | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """处理一条 JSON-RPC 消息；通知返回 None。"""
    if message.get("jsonrpc") != "2.0":
        return _error(message.get("id"), -32600, "jsonrpc 必须是 '2.0'。")
    method = message.get("method")
    message_id = message.get("id")
    params = message.get("params") or {}
    if not isinstance(method, str):
        return _error(message_id, -32600, "缺少 method。")
    if method == "notifications/initialized":
        return None
    if method == "initialize":
        return _result(
            message_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            },
        )
    if method == "ping":
        return _result(message_id, {})
    if method == "tools/list":
        return _result(message_id, {"tools": tools if tools is not None else TOOLS})
    if method == "tools/call":
        if not isinstance(params, dict):
            return _error(message_id, -32602, "params 必须是对象。")
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return _error(message_id, -32602, "tools/call 需要 name 与 arguments。")
        try:
            reply = _result(message_id, _call_tool(core, name, arguments))
        except KeyError:
            return _error(message_id, -32602, f"未知工具：{name}")
        except ToolError as error:
            reply = _result(message_id, _text_result({"error": str(error)}, is_error=True))
        except Exception as error:  # pragma: no cover - 防御未知崩溃
            return _error(message_id, -32603, f"工具内部错误：{error}")
        if audit is not None:
            result = reply.get("result") or {}
            audit.record(
                tool=name,
                arguments=arguments,
                ok=not bool(result.get("isError")),
                error=(
                    None
                    if not result.get("isError")
                    else json.loads(result["content"][0]["text"]).get("error")
                ),
            )
        return reply
    return _error(message_id, -32601, f"未知方法：{method}")


def serve_stdio(
    core: CognitiveCore,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    audit: AuditLog | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> None:
    """按行读 JSON-RPC；空行跳过，每条响应一行。"""
    source = stdin or sys.stdin
    sink = stdout or sys.stdout
    for line in source:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("消息必须是 JSON 对象。")
        except (json.JSONDecodeError, ValueError) as error:
            sink.write(json.dumps(_error(None, -32700, f"解析失败：{error}")) + "\n")
            sink.flush()
            continue
        response = handle_message(core, message, audit=audit, tools=tools)
        if response is not None:
            sink.write(json.dumps(response, ensure_ascii=False) + "\n")
            sink.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="biosnn-mcp",
        description="BioSNN-Plug phase-5 MCP server (stdio JSON-RPC; read-only unless --allow-write)",
    )
    parser.add_argument(
        "--allow-write",
        action="store_true",
        help="打开写工具（memory.write / memory.consolidate / trigger.arm / feed / disarm）",
    )
    parser.add_argument(
        "--audit-log",
        type=Path,
        default=None,
        help="把每次 tools/call 记成一行 JSONL 到这个路径",
    )
    args = parser.parse_args(argv)
    core = InMemoryCore(writes_enabled=args.allow_write)
    audit = AuditLog(path=args.audit_log)
    serve_stdio(core, audit=audit)
    return 0


def main_write(argv: list[str] | None = None) -> int:
    """写版本的入口：提议进队列，门控处置，落库到 biosnn 侧的持久化文件。

    与只读版共用同一份源码与同一个版本号（决策 1）；工具表只是多了四个，
    只读版的九个原样保留（W6）。
    """
    parser = argparse.ArgumentParser(
        prog="biosnn-mcp-write",
        description=(
            "BioSNN-Plug phase-5 MCP server, write mode "
            "(proposals -> consolidation gate -> persistent stand-in store)"
        ),
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="状态目录（默认取 $BIOSNN_MCP_STATE，否则 ~/.biosnn-mcp）",
    )
    parser.add_argument(
        "--store", type=Path, default=None, help="记忆持久化文件（默认 <state>/store.json）"
    )
    parser.add_argument(
        "--queue", type=Path, default=None, help="提议队列（默认 <state>/proposals.sqlite3）"
    )
    parser.add_argument(
        "--audit-log", type=Path, default=None, help="审计日志（默认 <state>/audit.jsonl）"
    )
    args = parser.parse_args(argv)

    state = args.state_dir or Path(
        os.environ.get("BIOSNN_MCP_STATE") or (Path.home() / ".biosnn-mcp")
    )
    store = args.store or state / "store.json"
    queue = args.queue or state / "proposals.sqlite3"
    audit = AuditLog(path=args.audit_log or state / "audit.jsonl")
    core = WriteCore(store, queue, audit=audit)
    serve_stdio(core, audit=audit, tools=write_tools())
    return 0
