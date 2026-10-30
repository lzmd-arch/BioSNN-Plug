"""P2：写版本 MCP 的端到端验收（W1–W6 的系统级跑法）。

走真入口：起一个 ``python -m biosnn_mcp write`` 子进程，用 stdin/stdout 说 JSON-RPC，
与真实 MCP 客户端看到的完全一样。库文件、审计文件、队列都在临时目录里，跑完即删。

    python scripts/e2e_write_mcp.py            # 默认写到系统临时目录
    python scripts/e2e_write_mcp.py --keep     # 保留目录，打印路径

退出码非零表示某条判据没过——CI 可以拿它当一条腿。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "packages" / "biosnn-mcp" / "src"

CHECKS: list[tuple[str, bool, str]] = []


def check(rule: str, ok: bool, detail: str) -> None:
    CHECKS.append((rule, bool(ok), detail))


class Session:
    """一个最小的 stdio MCP 客户端：一问一答，超时就报错而不是挂着。"""

    def __init__(self, state_dir: Path) -> None:
        env = dict(os.environ, PYTHONPATH=str(SRC))
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "biosnn_mcp", "write", "--state-dir", str(state_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=env,
        )
        self._id = 0

    def request(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        message = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            message["params"] = params
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            stderr = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"服务器没有响应：{stderr[:400]}")
        return json.loads(line)

    def call(self, name: str, arguments: dict | None = None) -> tuple[dict, bool]:
        reply = self.request("tools/call", {"name": name, "arguments": arguments or {}})
        payload = json.loads(reply["result"]["content"][0]["text"])
        return payload, bool(reply["result"].get("isError"))

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=10)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="保留临时目录")
    args = parser.parse_args(argv)

    state = Path(tempfile.mkdtemp(prefix="biosnn-mcp-e2e-"))
    store, audit = state / "store.json", state / "audit.jsonl"
    session = Session(state)
    try:
        session.request("initialize")
        tools = [tool["name"] for tool in session.request("tools/list")["result"]["tools"]]
        check(
            "W6 工具表 13 个且只读 9 个在前", len(tools) == 13 and "queue.list" in tools, str(tools)
        )

        # --- W1：提议 → 接受 → 可检索 → 三条审计
        accepted, _ = session.call(
            "propose.memory.write",
            {
                "text": "the trigger fired 3 times across 20 turns on the real stream",
                "proposal_id": "e2e-1",
                "source": "user",
            },
        )
        check(
            "W1 提议被判 accepted",
            accepted.get("status") == "accepted",
            json.dumps(accepted, ensure_ascii=False),
        )
        hits, _ = session.call("memory.search", {"query": "trigger"})
        check(
            "W1 落库后可检索", hits.get("count", 0) >= 1, json.dumps(hits, ensure_ascii=False)[:200]
        )

        events = [json.loads(line)["tool"] for line in audit.read_text("utf-8").splitlines()]
        needed = {"proposal.received", "gate.accepted", "memory.applied"}
        check("W1 审计三条齐全", needed <= set(events), str(sorted(set(events))))

        # --- W2：重复被拒、库不变
        before = sha256(store)
        duplicate, _ = session.call(
            "propose.memory.write",
            {
                "text": "the trigger fired 3 times across 20 turns on the real stream",
                "proposal_id": "e2e-dup",
                "source": "user",
            },
        )
        check(
            "W2 重复判 rejected",
            duplicate.get("status") == "rejected" and "duplicate" in duplicate.get("reason", ""),
            json.dumps(duplicate, ensure_ascii=False),
        )
        check("W2 库逐字节未变", sha256(store) == before, "")

        # --- W3：同一 proposal_id 只落一条
        replay, _ = session.call(
            "propose.memory.write",
            {
                "text": "the trigger fired 3 times across 20 turns on the real stream",
                "proposal_id": "e2e-1",
            },
        )
        check(
            "W3 重放同一 proposal_id 不产生新条目",
            replay.get("replayed") is True and replay.get("seq") == accepted.get("seq"),
            json.dumps(replay, ensure_ascii=False),
        )

        # --- W4：回滚后库逐字节回到写前
        before_second = sha256(store)
        second, _ = session.call(
            "propose.memory.write",
            {
                "text": "a second entry written only to be rolled back in this run",
                "proposal_id": "e2e-2",
                "source": "user",
            },
        )
        check(
            "W4 第二条 accepted",
            second.get("status") == "accepted",
            json.dumps(second, ensure_ascii=False),
        )
        reverted, error = session.call(
            "memory.revert", {"seq": second["seq"], "reason": "e2e rollback"}
        )
        check(
            "W4 撤销成功",
            not error and reverted.get("reverted") is True,
            json.dumps(reverted, ensure_ascii=False),
        )
        check("W4 库逐字节回到写前", sha256(store) == before_second, "")

        # --- W5：无旁路（对 MCP 客户端而言，写工具的输入里没有"直接落库"这条路）
        rejected, is_error = session.call("propose.memory.write", {"text": "missing proposal id"})
        check(
            "W5 缺少 proposal_id 时拒绝而不是落库",
            is_error and "proposal_id" in json.dumps(rejected, ensure_ascii=False),
            json.dumps(rejected, ensure_ascii=False),
        )

        # --- 门控可配置：把配额压到 0，下一条应转待审而不是被丢弃
        configured, _ = session.call(
            "gate.configure", {"reason": "e2e quota probe", "changes": {"quota_hourly": 0}}
        )
        check(
            "门控配置生效",
            configured.get("gate", {}).get("quota_hourly") == 0,
            json.dumps(configured, ensure_ascii=False),
        )
        held, _ = session.call(
            "propose.memory.write",
            {
                "text": "this one should be held because the hourly quota is zero",
                "proposal_id": "e2e-3",
                "source": "user",
            },
        )
        check(
            "超配额的提议转待审（不丢弃）",
            held.get("status") == "pending" and "quota" in held.get("reason", ""),
            json.dumps(held, ensure_ascii=False),
        )

        status, _ = session.call("queue.status")
        check(
            "队列统计自洽",
            status.get("counts", {}).get("accepted") == 1
            and status.get("counts", {}).get("reverted") == 1,
            json.dumps(status, ensure_ascii=False)[:300],
        )
    finally:
        session.close()

    width = max(len(rule) for rule, _, _ in CHECKS)
    for rule, ok, detail in CHECKS:
        print(f"{'PASS' if ok else 'FAIL'}  {rule.ljust(width)}  {detail[:120]}")
    failed = [rule for rule, ok, _ in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} 通过；状态目录：{state}")
    if not args.keep and not failed:
        import shutil

        shutil.rmtree(state, ignore_errors=True)
        print("（已清理临时目录；用 --keep 保留）")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
