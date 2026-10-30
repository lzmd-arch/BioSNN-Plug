"""M0 只读 MCP Server 的协议与工具测试。"""

from __future__ import annotations

import io
import json

from biosnn_mcp import (
    PROTOCOL_VERSION,
    TOOLS,
    InMemoryCore,
    MemoryEntry,
    handle_message,
    serve_stdio,
)


def _core() -> InMemoryCore:
    return InMemoryCore(
        [
            MemoryEntry("m1", "spiking working memory trace"),
            MemoryEntry("m2", "episodic memory paired associate"),
            MemoryEntry("m3", "liquid time constant dynamics"),
        ]
    )


class TestHandshake:
    def test_initialize_reports_tools_capability(self):
        reply = handle_message(_core(), {"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert reply["result"]["protocolVersion"] == PROTOCOL_VERSION
        assert reply["result"]["capabilities"] == {"tools": {}}
        assert reply["result"]["serverInfo"]["name"] == "biosnn-mcp"

    def test_initialized_notification_has_no_reply(self):
        assert (
            handle_message(_core(), {"jsonrpc": "2.0", "method": "notifications/initialized"})
            is None
        )

    def test_tools_list_exposes_read_only_and_write_tools(self):
        """M1 表面：四个只读 + 五个写（写工具在只读模式下会被拒，但必须可被发现）。"""
        reply = handle_message(_core(), {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = [tool["name"] for tool in reply["result"]["tools"]]
        assert names == [
            "state.query",
            "memory.search",
            "memory.write",
            "memory.consolidate",
            "trigger.status",
            "trigger.arm",
            "trigger.disarm",
            "trigger.feed",
            "trigger.poll",
        ]
        assert len(TOOLS) == 9


class TestTools:
    def test_state_query_is_read_only_and_llm_not_connected(self):
        reply = handle_message(
            _core(),
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "state.query", "arguments": {}},
            },
        )
        payload = json.loads(reply["result"]["content"][0]["text"])
        assert payload["read_only"] is True
        assert payload["llm"] == "not_connected"
        assert payload["autonomy"] == "disabled"
        assert payload["memory_entries"] == 3

    def test_memory_search_returns_ranked_hits(self):
        reply = handle_message(
            _core(),
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "memory.search",
                    "arguments": {"query": "episodic memory", "top_k": 2},
                },
            },
        )
        payload = json.loads(reply["result"]["content"][0]["text"])
        assert payload["count"] == 2
        assert payload["hits"][0]["id"] == "m2"
        assert payload["hits"][0]["score"] == 1.0

    def test_trigger_status_is_never_armed_in_m0(self):
        reply = handle_message(
            _core(),
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "trigger.status", "arguments": {}},
            },
        )
        payload = json.loads(reply["result"]["content"][0]["text"])
        assert payload["armed"] is False
        assert "read-only" in payload["reason"]

    def test_bad_tool_argument_is_tool_error_not_crash(self):
        reply = handle_message(
            _core(),
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {"name": "memory.search", "arguments": {"query": ""}},
            },
        )
        assert reply["result"]["isError"] is True

    def test_unknown_tool_is_protocol_error(self):
        reply = handle_message(
            _core(),
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {"name": "memory.delete", "arguments": {}},
            },
        )
        assert reply["error"]["code"] == -32602

    def test_unknown_method_is_protocol_error(self):
        reply = handle_message(_core(), {"jsonrpc": "2.0", "id": 8, "method": "resources/list"})
        assert reply["error"]["code"] == -32601


class TestStdio:
    def test_line_protocol_skips_blank_lines_and_notifications(self):
        source = io.StringIO(
            "\n"
            + json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
            + "\n"
            + json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
            + "\n"
            + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            + "\n"
        )
        sink = io.StringIO()
        serve_stdio(_core(), stdin=source, stdout=sink)
        replies = [json.loads(line) for line in sink.getvalue().splitlines()]
        assert len(replies) == 2
        assert replies[0]["id"] == 1
        assert replies[1]["id"] == 2

    def test_parse_error_returns_jsonrpc_error(self):
        sink = io.StringIO()
        serve_stdio(_core(), stdin=io.StringIO("not-json\n"), stdout=sink)
        reply = json.loads(sink.getvalue())
        assert reply["error"]["code"] == -32700
