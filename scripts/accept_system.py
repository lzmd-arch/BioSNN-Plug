"""P6：系统级验收（S2–S4），走真的 stdio 会话。

S1 的证据来自别处（干净 venv 装 wheel、CI 在 Linux 上跑同一条 smoke），这里不重复；
这个脚本只负责能在这个仓里跑完的那几条：

* S2：写版本的工具表与预期一致（13 个）；
* S3：合成一段「显著度上升 → 触发 → 提议 → 巩固 → 查回」，逐条断言，审计齐全；
* S4：撤销后库文件哈希回到写入前，审计里有撤销记录；
* 回归：只读版仍然拒绝写，且工具表与 0.2.0 相同。

    python scripts/accept_system.py [--keep]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "packages" / "biosnn-mcp" / "src"
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

RESULTS: list[tuple[str, bool, str]] = []


def check(rule: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((rule, bool(ok), detail))


class Session:
    def __init__(self, state_dir: Path, *, write_mode: bool) -> None:
        env = dict(os.environ, PYTHONPATH=str(SRC))
        argv = [sys.executable, "-m", "biosnn_mcp"]
        if write_mode:
            argv += ["write", "--state-dir", str(state_dir)]
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=env,
        )
        self._id = 0

    def _send(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        message: dict = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            message["params"] = params
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("服务器没有响应")
        return json.loads(line)

    def call(self, name: str, arguments: dict | None = None) -> tuple[dict, bool]:
        reply = self._send("tools/call", {"name": name, "arguments": arguments or {}})
        return json.loads(reply["result"]["content"][0]["text"]), bool(reply["result"].get("isError"))

    def tools(self) -> list[str]:
        return [tool["name"] for tool in self._send("tools/list")["result"]["tools"]]

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=10)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args(argv)

    state = Path(tempfile.mkdtemp(prefix="biosnn-accept-"))
    store, audit = state / "store.json", state / "audit.jsonl"
    session = Session(state, write_mode=True)
    try:
        # ── S2：工具表
        tools = session.tools()
        check("S2 写版本工具表 13 个", len(tools) == 13, str(tools))

        # ── S3：合成对话 → 触发 → 记忆 → 巩固 → 查回
        armed, _ = session.call("propose.trigger.arm", {"reason": "acceptance run"})
        check("S3 触发已武装", armed.get("armed") is True, json.dumps(armed, ensure_ascii=False))

        fired_at = None
        for step in range(1, 25):
            reply, _ = session.call("propose.trigger.feed", {"salience": min(1.0, step / 10)})
            if reply.get("fired"):
                fired_at = step
                break
        check("S3 显著度上升后触发", fired_at is not None, f"第 {fired_at} 步")

        events, _ = session.call("trigger.poll")
        check("S3 触发事件可取出", events.get("count", 0) >= 1, json.dumps(events, ensure_ascii=False)[:160])

        before = sha256(store) if store.exists() else ""
        written, _ = session.call(
            "propose.memory.write",
            {
                "text": "the trigger fired after 3 steps and the gate accepted this line",
                "proposal_id": "accept-1",
                "source": "user",
            },
        )
        check("S3 提议被判 accepted", written.get("status") == "accepted", json.dumps(written, ensure_ascii=False)[:200])

        hit, _ = session.call("memory.search", {"query": "trigger"})
        check("S3 落库后可查回", hit.get("count", 0) >= 1, json.dumps(hit, ensure_ascii=False)[:160])

        merged, _ = session.call(
            "propose.memory.consolidate", {"proposal_id": "accept-cons", "threshold": 0.0, "rounds": 1}
        )
        check("S3 巩固轮到 maintenance accept", merged.get("status") == "accepted", json.dumps(merged, ensure_ascii=False)[:160])

        events = [json.loads(line)["tool"] for line in audit.read_text("utf-8").splitlines()]
        needed = {"proposal.received", "gate.accepted", "memory.applied"}
        check("S3 审计三条齐全", needed <= set(events), str(sorted(set(events))))

        # ── S4：撤销 → 哈希回到写入前 → 审计里有撤销
        reverted, error = session.call(
            "memory.revert", {"seq": written["seq"], "reason": "acceptance rollback"}
        )
        check("S4 撤销成功", not error and reverted.get("reverted") is True, json.dumps(reverted, ensure_ascii=False)[:160])
        after = sha256(store) if store.exists() else ""
        check("S4 撤销后库文件哈希回到写前", after == before, f"{before[:12]}… → {after[:12]}…")
        events = [json.loads(line)["tool"] for line in audit.read_text("utf-8").splitlines()]
        check("S4 审计里有撤销记录", "memory.revert" in events, str(sorted(set(events))))
    finally:
        session.close()

    # ── 回归：只读版仍然拒绝写，工具表不变
    readonly = Session(state, write_mode=False)
    try:
        check("回归 只读版工具表与 0.2.0 相同", readonly.tools() == READ_ONLY_TOOLS, str(readonly.tools()))
        refused, is_error = readonly.call("propose.memory.write", {"text": "nope"})
        check(
            "回归 只读版拒绝写",
            is_error and "只读模式" in json.dumps(refused, ensure_ascii=False),
            json.dumps(refused, ensure_ascii=False)[:120],
        )
    finally:
        readonly.close()

    width = max(len(rule) for rule, _, _ in RESULTS)
    for rule, ok, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {rule.ljust(width)}  {detail[:110]}")
    failed = [rule for rule, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} 通过；状态目录：{state}")
    if not args.keep and not failed:
        shutil.rmtree(state, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
