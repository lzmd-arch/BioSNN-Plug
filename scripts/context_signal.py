"""**独占信号到底有没有预测力**——没有它，"真融合"就没有差集。

数据：真库 336 条记忆的 (created_at, content)。构造每个位置的上下文向量：
时段 sin/cos、星期 sin/cos、距上一条的间隔、是否同日、该条在库中的序号比例。

问题：**这些量能不能预测"下一个字是什么"，超过一元词频？**
- 若超过 ⇒ 独占信号存在，融合有差集，值得建语料、值得训；
- 若不超过 ⇒ 时间轴对文本没有预测力，融合需要的不是"更多时间特征"，是**别的模态**。

判据：留出集上，带上下文特征的分类器 top-1 **高于**一元先验 **≥ 1pp**。
"""

from __future__ import annotations

import json
import math
import sqlite3
import urllib.parse
from collections import Counter
from pathlib import Path

DB = Path("D:/AxoData/memory/mem.db")


def load() -> list[tuple[str, float]]:
    uri = "file:" + urllib.parse.quote(str(DB).replace("\\", "/")) + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        return [
            (r[0], float(r[1] or 0))
            for r in con.execute(
                "select content, created_at from memories where content is not null "
                "and length(content) > 20 order by created_at"
            )
        ]
    finally:
        con.close()


def features(ts: float, prev_ts: float, index: int, total: int) -> list[float]:
    day = ts / 86400.0
    hour = (ts % 86400) / 3600.0
    gap = 0.0 if prev_ts <= 0 else (ts - prev_ts) / 86400.0
    return [
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * (day % 7) / 7),
        math.cos(2 * math.pi * (day % 7) / 7),
        math.log1p(max(gap, 0.0)),
        float(gap < 1e-6),
        index / max(total, 1),
    ]


def main() -> int:
    rows = load()
    total = len(rows)
    print(f"记忆 {total} 条", flush=True)

    # 目标：每个位置的下一个汉字（一元基线：全库最常见字）
    pairs = []
    for i, (text, ts) in enumerate(rows):
        prev = rows[i - 1][1] if i else 0.0
        x = features(ts, prev, i, total)
        for j in range(len(text) - 1):
            pairs.append((x, text[j + 1]))
    cut = int(len(pairs) * 0.7)
    train, test = pairs[:cut], pairs[cut:]
    print(f"位置 {len(pairs)}（训练 {len(train)} / 留出 {len(test)}）", flush=True)

    counts: Counter[str] = Counter(t for _, t in train)
    prior = counts.most_common(1)[0][0]
    base = sum(1 for _, t in test if t == prior) / max(len(test), 1)

    # 带上下文的分类器：按上下文向量分桶，桶内取该桶训练集最常见字
    def bucket(x: list[float], bits: int = 3) -> tuple:
        return tuple(int((v + 1.0) * (2**bits - 1) / 2.0) for v in x)

    table: dict[tuple, Counter[str]] = {}
    for x, t in train:
        table.setdefault(bucket(x), Counter())[t] += 1
    hit = 0
    for x, t in test:
        dist = table.get(bucket(x))
        guess = dist.most_common(1)[0][0] if dist else prior
        hit += guess == t
    ctx = hit / max(len(test), 1)

    out = {
        "memories": total,
        "positions": len(pairs),
        "unigram_top1": round(base, 4),
        "context_top1": round(ctx, 4),
        "gain_pp": round((ctx - base) * 100, 2),
        "verdict": "时间轴有预测力" if (ctx - base) >= 0.01 else "时间轴没有可测的预测力",
        "note": "分桶法是最保守的下界；若连它都不涨，线性模型也不会涨。",
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    p = Path("sweep_results/context_signal/v1.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
