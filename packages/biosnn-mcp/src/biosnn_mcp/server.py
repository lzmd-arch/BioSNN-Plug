"""MCP Server：JSON-RPC 2.0 over stdio，无第三方依赖。

支持的方法：``initialize``、``notifications/initialized``、``ping``、``tools/list``、
``tools/call``。

工具面分两层：

* **只读**（任何模式都可用）：``state.query``、``memory.search``、``trigger.status``、
  ``trigger.poll``；
* **写**（服务器必须以 ``--allow-write`` 启动）：``memory.write``、``memory.consolidate``、
  ``trigger.arm``、``trigger.disarm``、``trigger.feed``。

默认只读是刻意的（ADR-0020 的 M0 边界）；``--audit-log`` 把每一次 ``tools/call`` 记成一行
JSONL（时间、工具、参数、成功与否），使调用可审计、可回放。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from .core import CognitiveCore, InMemoryCore, ToolError

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "biosnn-mcp", "version": "0.2.0"}

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
        "name": "memory.write",
        "description": "Store one memory entry. Requires a server started with --allow-write.",
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
        "name": "memory.consolidate",
        "description": "Run one sleep-consolidation pass. Requires --allow-write.",
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
        "name": "trigger.arm",
        "description": "Arm the spiking trigger; a reason string is required. Requires --allow-write.",
        "inputSchema": {
            "type": "object",
            "properties": {"reason": {"type": "string", "minLength": 1}},
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "trigger.disarm",
        "description": "Disarm the trigger and drop pending events. Requires --allow-write.",
        "inputSchema": _EMPTY_SCHEMA,
    },
    {
        "name": "trigger.feed",
        "description": "Advance the spiking trigger by one step with a salience in [0, 1]. Requires --allow-write.",
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


def _call_tool(core: CognitiveCore, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
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
    if name == "memory.write":
        _reject_unknown(arguments, {"text", "id", "tags"})
        tags = arguments.get("tags") or []
        if not isinstance(tags, list):
            raise ToolError("tags 必须是数组。")
        payload = core.write(
            arguments.get("text"),
            arguments.get("id"),
            tuple(tags),
        )
        return _text_result(payload)
    if name == "memory.consolidate":
        _reject_unknown(arguments, {"threshold", "rounds", "noise_sigma", "seed"})
        payload = core.consolidate(
            threshold=float(arguments.get("threshold", 0.95)),
            rounds=int(arguments.get("rounds", 3)),
            noise_sigma=float(arguments.get("noise_sigma", 0.05)),
            seed=int(arguments.get("seed", 0)),
        )
        return _text_result(payload)
    if name == "trigger.arm":
        _reject_unknown(arguments, {"reason"})
        return _text_result(core.arm(arguments.get("reason")))
    if name == "trigger.disarm":
        if arguments:
            raise ToolError("trigger.disarm 不接受参数。")
        return _text_result(core.disarm())
    if name == "trigger.feed":
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
        return _result(message_id, {"tools": TOOLS})
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
        response = handle_message(core, message, audit=audit)
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
