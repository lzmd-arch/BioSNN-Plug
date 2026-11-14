"""端到端离线评测（20 题）：**目前所有改动都没有下游判据**，这是第一个。

题目自动从记忆库构造：
- 取一条记忆，选两个不同的 3–4 字词 A（主题）与 B（事实）；
- 问「关于「A」，我说过什么？」；答对 = 回答里出现 **B**。

两个臂：裸千问（不给记忆）vs 带检索（top-3 注入）。
指标：答对率 / 拒答率 / 端到端延迟。
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from pathlib import Path

SLOW = "http://127.0.0.1:8081/v1/chat/completions"
DB = Path("D:/AxoData/memory/mem.db")
PERSONA = "你是千岁。优先使用注入给你的记忆；记忆里没有的事就说不知道，不要猜。"
REFUSAL = ("不知道", "没有记录", "没有关于", "没有直接说明", "查不到")


def post(payload: dict) -> dict:
    req = urllib.request.Request(
        SLOW,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=180) as handle:
        return json.loads(handle.read())


def memories() -> list[str]:
    uri = "file:" + urllib.parse.quote(str(DB).replace("\\", "/")) + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        return [
            r[0]
            for r in con.execute(
                "select content from memories where content is not null and length(content) > 30"
            )
        ]
    finally:
        con.close()


def terms(text: str) -> list[str]:
    return [t for t in re.findall(r"[\u4e00-\u9fff]{3,4}", text)]


def ask(question: str, evidence: str = "") -> tuple[str, float]:
    system = PERSONA + ("\n\n检索到的记忆：\n" + evidence if evidence else "")
    t0 = time.perf_counter()
    body = post(
        {
            "model": "qwen3",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": question + " /no_think"},
            ],
            "max_tokens": 100,
            "temperature": 0.0,
        }
    )
    return str(body["choices"][0]["message"]["content"]).strip(), time.perf_counter() - t0


def retrieve(q: str, mems: list[str], k: int = 3) -> list[str]:
    qt = set(terms(q))
    scored = []
    for m in mems:
        overlap = qt & set(terms(m))
        if overlap:
            scored.append((len(overlap), m))
    scored.sort(key=lambda x: -x[0])
    return [m for _, m in scored[:k]]


def main() -> int:
    mems = memories()
    rows = []
    for text in mems:
        ts = terms(text)
        if len(ts) < 4:
            continue
        topic, fact = ts[0], ts[-1]
        if topic == fact:
            continue
        q = f"关于「{topic}」，我说过什么？"
        hits = retrieve(q, mems)
        ev = "\n".join(f"- {m[:80]}" for m in hits)
        bare, t_bare = ask(q)
        with_ev, t_ev = ask(q, ev)
        rows.append(
            {
                "question": q,
                "expect": fact,
                "retrieved": len(hits),
                "bare_ok": fact in bare,
                "ev_ok": fact in with_ev,
                "bare_refused": any(r in bare for r in REFUSAL),
                "ev_refused": any(r in with_ev for r in REFUSAL),
                "t_bare": round(t_bare, 2),
                "t_ev": round(t_ev, 2),
            }
        )
        if len(rows) >= 20:
            break
        print(
            f"  {q[:22]:24s} 期望 {fact:6s} 裸={rows[-1]['bare_ok']} 带证据={rows[-1]['ev_ok']}",
            flush=True,
        )

    n = max(len(rows), 1)
    out = {
        "questions": len(rows),
        "answer_rate": {
            "bare": round(sum(r["bare_ok"] for r in rows) / n, 3),
            "with_evidence": round(sum(r["ev_ok"] for r in rows) / n, 3),
        },
        "refusal_rate": {
            "bare": round(sum(r["bare_refused"] for r in rows) / n, 3),
            "with_evidence": round(sum(r["ev_refused"] for r in rows) / n, 3),
        },
        "latency_s": {
            "bare": round(sum(r["t_bare"] for r in rows) / n, 2),
            "with_evidence": round(sum(r["t_ev"] for r in rows) / n, 2),
        },
    }
    print("\n" + json.dumps(out, ensure_ascii=False, indent=1))
    p = Path("sweep_results/e2e_offline/20q.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"summary": out, "rows": rows}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"结果 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
