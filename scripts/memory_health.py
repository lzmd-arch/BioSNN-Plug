"""记忆卫生诊断：**47 条从未被召回，是检索漏了，还是它们本就无关？**

做法：拿每条记忆自己的词构造一个查询，看它能不能把自己检索回 top-5。
- 回得来 ⇒ 检索没问题，那 47 条"从未召回"是**使用分布**的问题（没人问过相关的事）；
- 回不来 ⇒ **检索漏**，是缺陷。

同时统计权重分布，看 0.138 是稳态还是在下滑。
"""

from __future__ import annotations

import json
import re
import sqlite3
import urllib.parse
from pathlib import Path

DB = Path("D:/AxoData/memory/mem.db")


def terms(text: str) -> list[str]:
    return list(re.findall(r"[\u4e00-\u9fff]{2,4}|[A-Za-z]{3,}", text))


def main() -> int:
    uri = "file:" + urllib.parse.quote(str(DB).replace("\\", "/")) + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    rows = con.execute(
        "select memory_id, content, coalesce(access_count,0), coalesce(weight,0) "
        "from memories where content is not null and length(content) > 4"
    ).fetchall()
    con.close()
    print(f"记忆 {len(rows)} 条", flush=True)

    docs = [(mid, text, ac, w, set(terms(text))) for mid, text, ac, w in rows]
    never = [d for d in docs if d[2] == 0]
    print(f"从未被召回 {len(never)} 条", flush=True)

    found = 0
    missing = []
    for mid, text, _ac, _w, doc_terms in never:
        query = doc_terms
        # **自己也在候选里**——之前排除掉自己再查"自己在不在"，逻辑上永远为假。
        scored = sorted(
            ((len(query & other), oid) for oid, _t, _a, _w2, other in docs), reverse=True
        )[:5]
        if any(oid == mid for _, oid in scored):
            found += 1
        else:
            missing.append(mid)

    weights = sorted(d[3] for d in docs)
    out = {
        "memories": len(docs),
        "never_recalled": len(never),
        "self_retrievable_top5": found,
        "retrieval_miss": len(missing),
        "weight": {
            "min": round(weights[0], 4),
            "p25": round(weights[len(weights) // 4], 4),
            "median": round(weights[len(weights) // 2], 4),
            "p75": round(weights[3 * len(weights) // 4], 4),
            "max": round(weights[-1], 4),
        },
        "note": "自检索用同一套词法打分，只用于区分'检索漏'与'使用分布'。",
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    p = Path("sweep_results/memory_health/v1.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"结果 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
