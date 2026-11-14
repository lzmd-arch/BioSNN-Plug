"""端到端：用户**真实问过**的 40 条提问。

没有逐条正确答案的标注（只剩 support 28 / none 12 的聚合数），所以**不判对错**，
只量三件机器能判的事，并把 40 条回答导出成可读文件供人工复核：

  检索是否返回证据 / 是否拒答 / 端到端延迟

拒答率是这里最有用的一个数：40 条里用户真正关心的东西，她应该是答得出来的。
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
QS = Path("C:/Users/离中梦蝶/axo-annotation/questions.jsonl")
PERSONA = "你是千岁。优先使用注入给你的记忆；记忆里没有的事就说不知道，不要猜。"
REFUSAL = ("不知道", "没有记录", "没有关于", "没有直接说明", "查不到", "不清楚")


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
                "select content from memories where content is not null and length(content) > 4"
            )
        ]
    finally:
        con.close()


def terms(text: str) -> list[str]:
    return list(re.findall(r"[\u4e00-\u9fff]{2,4}|[A-Za-z]{3,}", text))


def retrieve(q: str, mems: list[str], k: int = 3) -> list[str]:
    qt = set(terms(q))
    scored = [(len(qt & set(terms(m))), m) for m in mems]
    scored = [x for x in scored if x[0] > 0]
    scored.sort(key=lambda x: -x[0])
    return [m for _, m in scored[:k]]


def answer(q: str, ev: str) -> tuple[str, float]:
    system = PERSONA + ("\n\n检索到的记忆：\n" + ev if ev else "")
    t0 = time.perf_counter()
    body = post(
        {
            "model": "qwen3",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": q + " /no_think"},
            ],
            "max_tokens": 120,
            "temperature": 0.0,
        }
    )
    return str(body["choices"][0]["message"]["content"]).strip(), time.perf_counter() - t0


def main() -> int:
    mems = memories()
    items = [json.loads(x) for x in QS.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = []
    for item in items:
        q = str(item.get("question") or "").strip()
        if not q:
            continue
        hits = retrieve(q, mems)
        ev = "\n".join(f"- {m[:80]}" for m in hits)
        text, dt = answer(q, ev)
        rows.append(
            {
                "id": item.get("id"),
                "q": q,
                "retrieved": len(hits),
                "answer": text,
                "refused": any(r in text for r in REFUSAL),
                "latency": round(dt, 2),
            }
        )
        print(f"  {q[:20]:22s} 检索{len(hits)} 拒答={rows[-1]['refused']} {dt:.2f}s", flush=True)

    n = max(len(rows), 1)
    out = {
        "questions": len(rows),
        "retrieval_hit_rate": round(sum(1 for r in rows if r["retrieved"]) / n, 3),
        "refusal_rate": round(sum(r["refused"] for r in rows) / n, 3),
        "latency_mean_s": round(sum(r["latency"] for r in rows) / n, 2),
    }
    print("\n" + json.dumps(out, ensure_ascii=False, indent=1))

    d = Path("sweep_results/e2e_offline")
    d.mkdir(parents=True, exist_ok=True)
    (d / "real40.json").write_text(
        json.dumps({"summary": out, "rows": rows}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    md = [
        "# 真实 40 条提问的端到端回答（供人工复核）",
        "",
        "| # | 提问 | 检索 | 她说的 |",
        "|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['id']} | {r['q'][:30]} | {r['retrieved']} | {r['answer'][:80].replace('|', '/')} |"
        )
    (d / "real40-review.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print(f"结果 {d / 'real40.json'}；复核表 {d / 'real40-review.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
