"""M1 写工具面的守卫：默认只读、显式武装、每次调用留审计。"""

from __future__ import annotations

import io
import json

import pytest
from biosnn_mcp import AuditLog, InMemoryCore, MemoryEntry, handle_message, serve_stdio


def _call(core, name, arguments=None, *, audit=None):
    reply = handle_message(
        core,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        },
        audit=audit,
    )
    return reply


def _payload(reply):
    return json.loads(reply["result"]["content"][0]["text"])


def _writable(entries=None):
    return InMemoryCore(entries, writes_enabled=True)


class TestReadOnlyGate:
    def test_write_is_refused_by_default(self):
        core = InMemoryCore()
        reply = _call(core, "propose.memory.write", {"text": "spike"})
        assert reply["result"]["isError"] is True
        assert "只读模式" in _payload(reply)["error"]
        assert len(core.search("spike", 5)) == 0

    def test_state_reports_the_two_milestones(self):
        readonly = _payload(_call(InMemoryCore(), "state.query"))
        writable = _payload(_call(_writable(), "state.query"))
        assert readonly["milestone"] == "M0" and readonly["read_only"] is True
        assert writable["milestone"] == "M1" and writable["read_only"] is False

    def test_arm_is_refused_without_allow_write(self):
        reply = _call(InMemoryCore(), "propose.trigger.arm", {"reason": "test"})
        assert reply["result"]["isError"] is True

    def test_poll_is_read_only_and_empty_when_nothing_fed(self):
        payload = _payload(_call(InMemoryCore(), "trigger.poll"))
        assert payload == {"events": [], "count": 0}


class TestWrites:
    def test_write_then_search_round_trip(self):
        core = _writable()
        payload = _payload(
            _call(core, "propose.memory.write", {"text": "liquid time constant", "tags": ["x"]})
        )
        assert payload["accepted"] is True and payload["id"] == "m0001"
        hits = _payload(_call(core, "memory.search", {"query": "liquid"}))["hits"]
        assert hits[0]["id"] == "m0001"
        assert hits[0]["tags"] == ["x"]

    def test_explicit_id_conflict_is_a_tool_error(self):
        core = _writable([MemoryEntry("m1", "first")])
        reply = _call(core, "propose.memory.write", {"text": "second", "id": "m1"})
        assert reply["result"]["isError"] is True
        assert "已存在" in _payload(reply)["error"]

    def test_write_rejects_unknown_arguments(self):
        reply = _call(_writable(), "propose.memory.write", {"text": "a", "colour": "red"})
        assert reply["result"]["isError"] is True
        assert "未知参数" in _payload(reply)["error"]

    def test_consolidate_merges_identical_texts(self):
        core = _writable(
            [
                MemoryEntry("a", "spiking working memory"),
                MemoryEntry("b", "spiking working memory"),
                MemoryEntry("c", "liquid time constant"),
            ]
        )
        payload = _payload(_call(core, "propose.memory.consolidate", {"threshold": 0.99}))
        assert payload["original_count"] == 3
        assert payload["final_count"] < payload["original_count"]
        assert payload["merged_pairs"]


class TestTriggerLifecycle:
    def test_arm_requires_a_reason(self):
        core = _writable()
        reply = _call(core, "propose.trigger.arm", {})
        assert reply["result"]["isError"] is True

    def test_arm_feed_poll_disarm(self):
        core = _writable()
        armed = _payload(_call(core, "propose.trigger.arm", {"reason": "manual test"}))
        assert armed["armed"] is True and armed["reason"] == "manual test"

        fired = 0
        for _ in range(200):
            payload = _payload(_call(core, "propose.trigger.feed", {"salience": 1.0}))
            fired += int(payload["fired"])
        assert fired > 0, "满显著性喂 200 步应当至少触发一次"
        events = _payload(_call(core, "trigger.poll"))
        assert events["count"] == fired
        assert _payload(_call(core, "trigger.poll"))["count"] == 0

        disarmed = _payload(_call(core, "propose.trigger.disarm"))
        assert disarmed["armed"] is False

    def test_feed_without_arming_queues_nothing(self):
        core = _writable()
        fired = 0
        for _ in range(200):
            fired += int(_payload(_call(core, "propose.trigger.feed", {"salience": 1.0}))["fired"])
        assert fired > 0
        assert _payload(_call(core, "trigger.poll"))["count"] == 0

    def test_feed_rejects_out_of_range_salience(self):
        reply = _call(_writable(), "propose.trigger.feed", {"salience": 1.5})
        assert reply["result"]["isError"] is True


class TestAudit:
    def test_every_call_is_recorded_ok_or_error(self):
        core = _writable()
        audit = AuditLog()
        _call(core, "propose.memory.write", {"text": "alpha"}, audit=audit)
        _call(core, "memory.search", {"query": ""}, audit=audit)
        assert [entry["tool"] for entry in audit.entries] == [
            "propose.memory.write",
            "memory.search",
        ]
        assert audit.entries[0]["ok"] is True
        assert audit.entries[1]["ok"] is False
        assert "query" in (audit.entries[1]["error"] or "")

    def test_audit_file_is_jsonl(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        audit = AuditLog(path=path)
        _call(_writable(), "state.query", audit=audit)
        _call(_writable(), "propose.memory.write", {}, audit=audit)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        decoded = [json.loads(line) for line in lines]
        assert decoded[0]["tool"] == "state.query" and decoded[0]["ok"] is True
        assert decoded[1]["ok"] is False


class TestStdio:
    def test_write_refused_through_the_line_protocol(self):
        lines = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "tools/call",
                "params": {"name": "propose.memory.write", "arguments": {"text": "x"}},
            }
        )
        sink = io.StringIO()
        serve_stdio(InMemoryCore(), stdin=io.StringIO(lines + "\n"), stdout=sink)
        reply = json.loads(sink.getvalue().strip())
        assert reply["result"]["isError"] is True


@pytest.mark.parametrize(
    "name",
    [
        "propose.memory.write",
        "propose.memory.consolidate",
        "propose.trigger.arm",
        "propose.trigger.disarm",
        "propose.trigger.feed",
    ],
)
def test_all_write_tools_are_refused_in_read_only_mode(name):
    arguments = {
        "propose.memory.write": {"text": "x"},
        "propose.memory.consolidate": {},
        "propose.trigger.arm": {"reason": "r"},
        "propose.trigger.disarm": {},
        "propose.trigger.feed": {"salience": 0.5},
    }[name]
    reply = _call(InMemoryCore(), name, arguments)
    assert reply["result"]["isError"] is True
