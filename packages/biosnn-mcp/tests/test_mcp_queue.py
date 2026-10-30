"""写版本（``biosnn-mcp-write``）：队列 → 门控 → 落库 → 可回滚。

这里逐条钉住计划书 §3.4 的 W1–W6。判据是**预注册**的，改动要写 amend——
所以测试名里带着编号，改判据时一眼能看出动了哪一条。
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path

import pytest
from biosnn_mcp import (
    TOOLS,
    AuditLog,
    Gate,
    GateConfig,
    InMemoryCore,
    ProposalQueue,
    ToolError,
    WriteCore,
    handle_message,
    serve_stdio,
    write_tools,
)
from biosnn_mcp.proposals import salience_score


def _core(tmp_path: Path, **kwargs) -> WriteCore:
    audit = kwargs.pop("audit", None)
    return WriteCore(tmp_path / "store.json", tmp_path / "queue.sqlite3", audit=audit, **kwargs)


def _call(core, name, arguments=None, *, audit=None, tools=None):
    return handle_message(
        core,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        },
        audit=audit,
        tools=tools,
    )


def _payload(reply):
    return json.loads(reply["result"]["content"][0]["text"])


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _propose(core, text, *, pid="p1", source="user", identifier=None, tags=()):
    return _payload(
        _call(
            core,
            "propose.memory.write",
            {
                "text": text,
                "proposal_id": pid,
                "source": source,
                "id": identifier,
                "tags": list(tags),
            },
        )
    )


class TestGateArithmetic:
    """显著度是常量算出来的，不是临场调的。"""

    def test_user_proposal_clears_the_bar(self):
        score = salience_score("the gate rejected 12 proposals today at 09:00", source="user")
        assert score >= GateConfig().accept_salience

    def test_llm_inference_scores_lower_than_a_user_statement(self):
        body = "the scheduler fired once per conversation in 20 trials"
        assert salience_score(body, source="user") > salience_score(body, source="llm")

    def test_duplicate_rejects_before_salience_is_considered(self):
        decision = Gate().evaluate(
            "anything at all, long enough to be salient",
            source="user",
            max_cosine=0.95,
            quota_used_hour=0,
            quota_used_day=0,
        )
        assert decision.status == "rejected"
        assert "duplicate" in decision.reason

    def test_low_salience_is_rejected_not_pended(self):
        # llm 来源 + 太短 + 与已有条目很像 ⇒ 0.15 + 0 + 0.05 + 0 = 0.20 < 0.30
        decision = Gate().evaluate(
            "ok", source="llm", max_cosine=0.8, quota_used_hour=0, quota_used_day=0
        )
        assert decision.status == "rejected"
        assert "not salient enough" in decision.reason

    def test_middle_band_is_pending(self):
        # llm + 太短：0.15 + 0 + 0.25 + 0 = 0.40，落在 0.30–0.60
        decision = Gate().evaluate(
            "ok", source="llm", max_cosine=0.0, quota_used_hour=0, quota_used_day=0
        )
        assert decision.status == "pending"
        assert "held for review" in decision.reason

    def test_novel_in_band_llm_text_sits_exactly_on_the_bar(self):
        # 这条是刻意写下来的边界：0.15 + 0.20 + 0.25 = 0.60，正好等于 accept 阈值
        assert salience_score("a short note about memory", source="llm") == pytest.approx(
            GateConfig().accept_salience
        )

    def test_quota_exhaustion_holds_instead_of_dropping(self):
        text = "the gate accepted 20 proposals in the last hour"
        decision = Gate(GateConfig(quota_hourly=1)).evaluate(
            text, source="user", max_cosine=0.0, quota_used_hour=1, quota_used_day=1
        )
        assert decision.status == "pending"
        assert "quota exhausted" in decision.reason


class TestW1EndToEnd:
    def test_proposal_accepted_searchable_and_audited_three_times(self, tmp_path):
        audit = AuditLog(path=tmp_path / "audit.jsonl")
        core = _core(tmp_path, audit=audit)
        payload = _propose(core, "the trigger fired 3 times in 20 turns", pid="w1")

        assert payload["status"] == "accepted", payload
        hits = _payload(_call(core, "memory.search", {"query": "trigger"}))["hits"]
        assert hits and "trigger" in hits[0]["text"]

        events = [
            json.loads(line)["tool"]
            for line in (tmp_path / "audit.jsonl").read_text("utf-8").splitlines()
        ]
        assert "proposal.received" in events
        assert "gate.accepted" in events
        assert "memory.applied" in events

    def test_state_reports_the_gate_scope(self, tmp_path):
        payload = _payload(_call(_core(tmp_path), "state.query"))
        assert payload["write_scope"] == "gate"
        assert payload["read_only"] is False
        assert payload["queue"]["accepted"] == 0


class TestW2Rejection:
    def test_duplicate_is_rejected_and_the_store_is_untouched(self, tmp_path):
        core = _core(tmp_path)
        first = _propose(core, "the gate rejected 12 proposals today at 09:00", pid="a")
        assert first["status"] == "accepted"
        before = _digest(tmp_path / "store.json")

        second = _propose(core, "the gate rejected 12 proposals today at 09:00", pid="b")
        assert second["status"] == "rejected"
        assert "duplicate" in second["reason"]
        assert _digest(tmp_path / "store.json") == before

    def test_identifier_clash_is_rejected_with_a_readable_reason(self, tmp_path):
        core = _core(tmp_path)
        _propose(core, "first entry about the arbitration sweep", pid="a", identifier="m0001")
        clash = _propose(
            core, "second entry about the arbitration sweep", pid="b", identifier="m0001"
        )
        assert clash["status"] == "rejected"
        assert "already exists" in clash["reason"]


class TestW3Idempotency:
    def test_same_proposal_id_lands_once(self, tmp_path):
        core = _core(tmp_path)
        first = _propose(core, "a memorable line of 24 chars", pid="same")
        second = _propose(core, "a memorable line of 24 chars", pid="same")
        assert first["seq"] == second["seq"]
        assert second["replayed"] is True
        assert second["entries"] == first["entries"] == 1
        assert core.queue.counts()["accepted"] == 1


class TestW4Revert:
    def test_revert_restores_the_store_byte_for_byte(self, tmp_path):
        core = _core(tmp_path)
        _propose(core, "baseline entry written before the test", pid="base")
        before = (tmp_path / "store.json").read_bytes()

        added = _propose(core, "another entry that will be rolled back", pid="temp")
        assert added["status"] == "accepted"
        assert (tmp_path / "store.json").read_bytes() != before

        payload = _payload(
            _call(core, "memory.revert", {"seq": added["seq"], "reason": "test rollback"})
        )
        assert payload["reverted"] is True
        assert (tmp_path / "store.json").read_bytes() == before
        assert len(core.search("rolled back", 5)) == 0

    def test_revert_twice_is_refused(self, tmp_path):
        core = _core(tmp_path)
        added = _propose(core, "an entry to be reverted exactly once", pid="once")
        _call(core, "memory.revert", {"seq": added["seq"], "reason": "first"})
        again = _call(core, "memory.revert", {"seq": added["seq"], "reason": "second"})
        assert again["result"]["isError"] is True
        assert "撤销过" in _payload(again)["error"]

    def test_pending_proposal_cannot_be_reverted(self, tmp_path):
        core = _core(tmp_path)
        held = _propose(core, "short note", pid="held", source="llm")
        assert held["status"] == "pending"
        reply = _call(core, "memory.revert", {"seq": held["seq"], "reason": "nope"})
        assert reply["result"]["isError"] is True


class TestW5NoBypass:
    def test_write_core_refuses_direct_writes(self, tmp_path):
        core = _core(tmp_path)
        with pytest.raises(ToolError, match="propose"):
            core.write("bypass attempt")
        with pytest.raises(ToolError, match="propose"):
            core.consolidate()

    def test_only_the_gate_touches_the_store(self):
        """AST 守卫：``self._store.write`` 只能出现在 ``_apply`` 里。"""
        source = Path(__file__).resolve().parents[1] / "src" / "biosnn_mcp" / "writecore.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        callers: set[str] = set()

        def is_store_call(node: ast.AST) -> bool:
            if not isinstance(node, ast.Call):
                return False
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr not in {"write", "consolidate"}:
                return False
            inner = func.value
            return (
                isinstance(inner, ast.Attribute)
                and inner.attr == "_store"
                and isinstance(inner.value, ast.Name)
                and inner.value.id == "self"
            )

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for child in ast.walk(node):
                    if is_store_call(child):
                        callers.add(node.name)
        assert callers == {"_apply"}, callers


READ_ONLY_TOOLS = [
    "state.query",
    "memory.search",
    "propose.memory.write",
    "propose.memory.consolidate",
    "trigger.status",
    "propose.trigger.arm",
    "propose.trigger.disarm",
    "propose.trigger.feed",
    "trigger.poll",
]


class TestW6ReadOnlyRegression:
    def test_tool_table_is_frozen(self):
        assert [tool["name"] for tool in TOOLS] == READ_ONLY_TOOLS

    def test_write_version_keeps_all_nine_and_adds_four(self):
        names = [tool["name"] for tool in write_tools()]
        assert names[:9] == READ_ONLY_TOOLS
        assert names[9:] == ["queue.status", "queue.list", "memory.revert", "gate.configure"]

    def test_read_only_refusal_semantics_are_unchanged(self):
        reply = _call(InMemoryCore(), "propose.memory.write", {"text": "x"})
        assert reply["result"]["isError"] is True
        assert "只读模式" in _payload(reply)["error"]

    def test_read_only_server_does_not_know_the_queue_tools(self):
        reply = _call(InMemoryCore(), "queue.status")
        assert reply["error"]["code"] == -32602
        assert "未知工具" in reply["error"]["message"]

    def test_audit_line_keeps_the_0_2_0_shape(self, tmp_path):
        audit = AuditLog(path=tmp_path / "audit.jsonl")
        core = _core(tmp_path, audit=audit)
        _propose(core, "an entry that produces an audit line", pid="audit")
        line = json.loads((tmp_path / "audit.jsonl").read_text("utf-8").splitlines()[0])
        assert set(line) == {"ts", "tool", "ok", "error", "arguments"}


class TestQueueSurface:
    def test_queue_list_and_status(self, tmp_path):
        core = _core(tmp_path)
        _propose(core, "the first entry in the queue listing", pid="l1")
        _propose(core, "ok", pid="l2", source="llm")
        status = _payload(_call(core, "queue.status"))
        assert status["counts"]["accepted"] == 1
        assert status["counts"]["pending"] == 1
        listing = _payload(_call(core, "queue.list", {"status": "accepted"}))
        assert listing["count"] == 1
        assert listing["proposals"][0]["proposal_id"] == "l1"

    def test_queue_list_rejects_unknown_status(self, tmp_path):
        reply = _call(_core(tmp_path), "queue.list", {"status": "whatever"})
        assert reply["result"]["isError"] is True

    def test_gate_configure_requires_a_reason_and_audits(self, tmp_path):
        audit = AuditLog(path=tmp_path / "audit.jsonl")
        core = _core(tmp_path, audit=audit)
        reply = _call(
            core, "gate.configure", {"reason": "tighter test", "changes": {"accept_salience": 0.8}}
        )
        assert _payload(reply)["gate"]["accept_salience"] == pytest.approx(0.8)
        assert core.gate.config.accept_salience == pytest.approx(0.8)
        tools = [
            json.loads(line)["tool"]
            for line in (tmp_path / "audit.jsonl").read_text("utf-8").splitlines()
        ]
        assert "gate.configure" in tools
        # 配置要能活过重启
        again = WriteCore(tmp_path / "store.json", tmp_path / "queue.sqlite3")
        assert again.gate.config.accept_salience == pytest.approx(0.8)

    def test_gate_configure_validates_ranges(self, tmp_path):
        reply = _call(
            _core(tmp_path),
            "gate.configure",
            {"reason": "bad", "changes": {"floor_salience": 0.9, "accept_salience": 0.5}},
        )
        assert reply["result"]["isError"] is True

    def test_consolidation_is_a_maintenance_accept_and_revertible(self, tmp_path):
        core = _core(tmp_path)
        _propose(core, "alpha beta gamma delta text for merging", pid="c1", identifier="a")
        _propose(core, "alpha beta gamma delta text for merging two", pid="c2", identifier="b")
        before = (tmp_path / "store.json").read_bytes()
        payload = _payload(
            _call(
                core,
                "propose.memory.consolidate",
                {"proposal_id": "cons-1", "threshold": 0.0, "rounds": 1, "noise_sigma": 0.0},
            )
        )
        assert payload["status"] == "accepted"
        _call(core, "memory.revert", {"seq": payload["seq"], "reason": "undo merge"})
        assert (tmp_path / "store.json").read_bytes() == before


class TestPersistence:
    def test_entries_survive_a_restart(self, tmp_path):
        core = _core(tmp_path)
        _propose(core, "a durable entry that must survive a restart", pid="d1")
        again = _core(tmp_path)
        assert len(again.search("durable", 5)) == 1
        assert again.queue.counts()["accepted"] == 1

    def test_stdio_session_round_trip(self, tmp_path):
        core = _core(tmp_path)
        stdin = io.StringIO(
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
            + "\n"
            + json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "propose.memory.write",
                        "arguments": {"text": "stdio entry about the queue", "proposal_id": "s1"},
                    },
                }
            )
            + "\n"
        )
        stdout = io.StringIO()
        serve_stdio(core, stdin=stdin, stdout=stdout, tools=write_tools())
        lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
        assert json.loads(lines[1]["result"]["content"][0]["text"])["status"] == "accepted"


class TestProposalQueueUnit:
    def test_enqueue_is_idempotent(self, tmp_path):
        queue = ProposalQueue(tmp_path / "q.sqlite3")
        first, created = queue.enqueue("x", "memory.write", {"text": "a"})
        second, created_again = queue.enqueue("x", "memory.write", {"text": "b"})
        assert created is True and created_again is False
        assert first == second
        assert queue.get(first)["payload"]["text"] == "a"

    def test_counts_and_quota_window(self, tmp_path):
        queue = ProposalQueue(tmp_path / "q.sqlite3")
        seq, _ = queue.enqueue("a", "memory.write", {"text": "t"})
        queue.decide(seq, "accepted", "fine")
        hour, day = queue.quota_usage()
        assert (hour, day) == (1, 1)
