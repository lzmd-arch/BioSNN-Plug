"""把 Axo 的真实会话导出成 (user, assistant) 对，供 R0 用。

**数据不出本机**：只读 `D:/AxoData/sessions/*.jsonl`，只写 `D:/AxoData/sel/lm/v1/real.jsonl`，
两边都在仓外。仓库里只有这个脚本。

口径（写死）：
- 只取**紧邻**的 user → assistant，且 assistant 内容非空；
- 跳过带 `tool_calls` 的 assistant、跳过 `tool` 角色、跳过 `proactive` 消息；
- 按**会话**记录，供后续按会话切分。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import pairwise
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_SESSIONS = Path("D:/AxoData/sessions")
DEFAULT_OUT = Path("D:/AxoData/sel/lm/v1/real.jsonl")


def load_pairs(sessions_dir: Path) -> tuple[list[dict], Counter]:
    stats: Counter = Counter()
    pairs: list[dict] = []
    for path in sorted(sessions_dir.glob("*.jsonl")):
        lines = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not lines or "session_id" not in lines[0]:
            stats["bad_file"] += 1
            continue
        header = lines[0]
        messages = lines[1:]
        stats["messages"] += len(messages)
        for previous, current in pairwise(messages):
            if previous.get("role") != "user" or current.get("role") != "assistant":
                continue
            if current.get("tool_calls") or previous.get("tool_calls"):
                stats["skip_tool_calls"] += 1
                continue
            if current.get("proactive"):
                stats["skip_proactive"] += 1
                continue
            user = (previous.get("content") or "").strip()
            assistant = (current.get("content") or "").strip()
            if not user or not assistant:
                stats["skip_empty"] += 1
                continue
            pairs.append(
                {
                    "session": header["session_id"],
                    "model": header.get("model_ref", ""),
                    "user": user,
                    "assistant": assistant,
                }
            )
            stats["pairs"] += 1
    return pairs, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出真实会话对")
    parser.add_argument("--sessions", type=Path, default=DEFAULT_SESSIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    pairs, stats = load_pairs(args.sessions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pairs),
        encoding="utf-8",
        newline="\n",
    )
    total_chars = sum(len(row["user"]) + len(row["assistant"]) for row in pairs)
    unique = {ch for row in pairs for ch in row["user"] + row["assistant"]}
    print(
        f"会话 {stats['bad_file'] + len({row['session'] for row in pairs})} 个，消息 {stats['messages']} 条"
    )
    print(
        f"**配对 {stats['pairs']} 对**（跳过 tool_calls {stats['skip_tool_calls']}、"
        f"proactive {stats['skip_proactive']}、空 {stats['skip_empty']}）"
    )
    print(f"总字符 {total_chars}，唯一字符 {len(unique)}")
    print(f"写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
