"""用**有界的膜电位式衰减**替代无界的指数衰减。

现状：`weight = clip01(w · exp(−ln2·Δ/half_life))` —— 没有底，真库中位数 ≈1e-21，
且更新因子 <1 ⇒ 被重新召回也回不来。

替代：`v ← floor + (v − floor)·exp(−Δ/τ)`，**下限 floor 是渐近线而不是夹取**，
所以它不会"掉到底然后卡住"，而是长期停在 floor 附近、被一次访问就能拉起来。

在真库的时间戳上对比两种动力学。
"""

from __future__ import annotations

import json
import math
import sqlite3
import urllib.parse
from pathlib import Path

import numpy as np

DB = Path("D:/AxoData/memory/mem.db")
HALF_LIFE_DAYS = 30.0  # 与 tier 的默认半衰期同量级
FLOOR = 0.05  # 渐近线
TAU_DAYS = 30.0


def load():
    uri = "file:" + urllib.parse.quote(str(DB).replace("\\", "/")) + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        return con.execute(
            "select coalesce(weight,0), coalesce(last_access_ts, created_at, 0), "
            "coalesce(created_at,0) from memories where content is not null"
        ).fetchall()
    finally:
        con.close()


def main() -> int:
    rows = load()
    now = max(r[2] for r in rows) + 3 * 86400.0  # 以库内最新时刻为"现在"
    cur, expo, bound = [], [], []
    for w, last_access, _created in rows:
        delta_days = max(now - float(last_access), 0.0) / 86400.0
        cur.append(float(w))
        expo.append(
            max(min(float(w) * math.exp(-math.log(2) * delta_days / HALF_LIFE_DAYS), 1.0), 0.0)
        )
        # 有界：渐近到 FLOOR，且被访问时能立刻回到 1
        v = FLOOR + (1.0 - FLOOR) * math.exp(-delta_days / TAU_DAYS)
        bound.append(v)

    def stats(xs):
        a = np.array(xs)
        return {
            "median": round(float(np.median(a)), 4),
            "p25": round(float(np.percentile(a, 25)), 4),
            "min": round(float(a.min()), 6),
            "below_1e-6": int((a < 1e-6).sum()),
        }

    out = {
        "memories": len(rows),
        "current": stats(cur),
        "exponential": stats(expo),
        "bounded": stats(bound),
        "note": "同一批时间戳、同一 half_life；差别只在有没有渐近下限。",
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    p = Path("sweep_results/memory_health/recency_bounded.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
