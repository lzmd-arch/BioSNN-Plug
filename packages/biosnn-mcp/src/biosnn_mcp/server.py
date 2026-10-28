"""M0 只读 MCP Server：JSON-RPC 2.0 over stdio，无第三方依赖。

支持的方法：``initialize``、``notifications/initialized``、``ping``、``tools/list``、
``tools/call``。工具只有三个只读项：``state.query``、``memory.search``、
``trigger.status``。没有 LLM、网络或写操作。
"""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from .core import CognitiveCore, InMemoryCore, ToolError

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "biosnn-mcp", "version": "0.1.0"}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "state.query",
        "description": "Read the phase-5 M0 cognitive-core status (read-only).",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "memory.search",
        "description": "Search the read-only in-memory episodic index by token overlap.",
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
        "name": "trigger.status",
        "description": "Read whether autonomous LLM triggering is armed (M0: never).",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]


def _error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}


def _result(message_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _text_result(payload: dict[str, Any], *, is_error: bool = False) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
        "isError": is_error,
    }


def _call_tool(core: CognitiveCore, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "state.query":
        if arguments:
            raise ToolError("state.query 不接受参数。")
        return _text_result(core.state())
    if name == "trigger.status":
        if arguments:
            raise ToolError("trigger.status 不接受参数。")
        return _text_result(core.trigger_status())
    if name == "memory.search":
        unknown = set(arguments) - {"query", "top_k"}
        if unknown:
            raise ToolError(f"memory.search 收到未知参数：{sorted(unknown)}")
        hits = core.search(arguments.get("query"), arguments.get("top_k", 5))
        return _text_result({"hits": hits, "count": len(hits)})
    raise KeyError(name)


def handle_message(core: CognitiveCore, message: dict[str, Any]) -> dict[str, Any] | None:
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
            return _result(message_id, _call_tool(core, name, arguments))
        except KeyError:
            return _error(message_id, -32602, f"未知工具：{name}")
        except ToolError as error:
            return _result(message_id, _text_result({"error": str(error)}, is_error=True))
        except Exception as error:  # pragma: no cover - 防御未知崩溃
            return _error(message_id, -32603, f"工具内部错误：{error}")
    return _error(message_id, -32601, f"未知方法：{method}")


def serve_stdio(
    core: CognitiveCore,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
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
        response = handle_message(core, message)
        if response is not None:
            sink.write(json.dumps(response, ensure_ascii=False) + "\n")
            sink.flush()


def main() -> int:
    serve_stdio(InMemoryCore())
    return 0
