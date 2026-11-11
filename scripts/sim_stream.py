"""模拟流：把 208 条记忆切成 N 天的"新增"，逐日建索引并评测。

**这个脚本只建基础设施，不证明 U-3。** 两件事必须写在前面：

1. **它测的是 U-1（换掉快系统），不是 U-3（冻结快系统）。** 两臂分别是
   "SNN 编码" 与 "字符 bigram 编码"，都没有在流上更新；
2. **U-3 不可能在主干上测**：一个人的一天 ≈ 10⁴ token，而 0.5B 权重需要 10⁹ token
   才动得起来——**差五个数量级**。U-3 只能落在小头（重排权重 / 风格头）或索引统计上。

所以本脚本的产出是**曲线**：随着索引进来，两路的 hit@K 怎么变。它为后续
"带漂移的流 + 小头在线更新"提供同一套脚手架。

用法::

    uv run python scripts/sim_stream.py --days 8 --host http://127.0.0.1:18734
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GOLD = Path("D:/Projects/BioSNN-MemoryTerminal/benchmarks/gold")


def _post(url: str, payload: dict | None, *, timeout: float) -> dict:
    """只用标准库。本机地址，显式绕过代理（与 Axo 那条同口径）。"""
    import urllib.request

    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data else "GET",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as handle:
        return json.loads(handle.read())


def lexical_matrix(texts: list[str]) -> np.ndarray:
    """字符 bigram 的稀疏计数，L2 归一化——不需要模型、不需要网络。"""
    index: dict[str, int] = {}
    rows: list[dict[int, float]] = []
    for text in texts:
        grams = [text[i : i + 2] for i in range(max(len(text) - 1, 1))]
        row: dict[int, float] = {}
        for gram in grams:
            slot = index.setdefault(gram, len(index))
            row[slot] = row.get(slot, 0.0) + 1.0
        rows.append(row)
    dense = np.zeros((len(texts), max(len(index), 1)), dtype=np.float32)
    for r, row in enumerate(rows):
        for slot, value in row.items():
            dense[r, slot] = value
    norms = np.linalg.norm(dense, axis=1, keepdims=True).clip(min=1e-12)
    return dense / norms


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="模拟流：逐日建索引并评测")
    ap.add_argument("--days", type=int, default=8)
    ap.add_argument("--host", default="http://127.0.0.1:18734")
    ap.add_argument("--gold", type=Path, default=GOLD)
    args = ap.parse_args(argv)

    memories = [
        json.loads(x) for x in (args.gold / "memories.jsonl").read_text("utf-8").splitlines()
    ]
    queries = [json.loads(x) for x in (args.gold / "queries.jsonl").read_text("utf-8").splitlines()]
    queries = [q for q in queries if q.get("expect")]
    print(f"记忆 {len(memories)} 条 / 查询 {len(queries)} 条，切成 {args.days} 天", flush=True)

    contents = [m["content"] for m in memories]
    health = _post(args.host + "/health", None, timeout=30)
    print(f"SNN 服务：{health['space_id']}", flush=True)
    snn = np.asarray(
        _post(args.host + "/encode", {"texts": contents}, timeout=300)["vectors"], dtype=np.float32
    )
    query_snn = np.asarray(
        _post(args.host + "/encode", {"texts": [q["query"] for q in queries]}, timeout=300)[
            "vectors"
        ],
        dtype=np.float32,
    )
    # 查询的词法向量要在**同一套 bigram 词表**里算，所以把记忆和查询拼起来建表再切回
    combined = lexical_matrix(contents + [q["query"] for q in queries])
    lex, query_lex = combined[: len(contents)], combined[len(contents) :]

    per_day = max(1, len(memories) // args.days)
    print(
        f"\n{'天':>3} {'入索引':>7} {'SNN hit@1':>10} {'词法 hit@1':>11} {'SNN hit@5':>10} {'词法 hit@5':>11}"
    )
    rows = []
    for day in range(1, args.days + 1):
        upto = min(per_day * day, len(memories))
        table_snn, table_lex = snn[:upto], lex[:upto]
        ids = [m["id"] for m in memories[:upto]]
        snn1 = lex1 = snn5 = lex5 = 0
        for query, sv, lv in zip(queries, query_snn, query_lex, strict=True):
            want = set(query["expect"])
            for table, vec, tag in ((table_snn, sv, "snn"), (table_lex, lv, "lex")):
                order = np.argsort(-(table @ vec))
                # **`rank` 是名次、`order[rank]` 才是记忆下标。**
                # 早先写成 `next(r for r in order ...)`，把下标当成了名次——
                # 于是 `position == 0` 变成"第 0 条记忆是不是答案"，实测只给出 1/77 的假数。
                position = next(
                    (rank for rank, idx in enumerate(order.tolist()) if ids[idx] in want), None
                )
                if tag == "snn":
                    snn1 += int(position == 0)
                    snn5 += int(position is not None and position < 5)
                else:
                    lex1 += int(position == 0)
                    lex5 += int(position is not None and position < 5)
        n = len(queries)
        rows.append(
            {
                "day": day,
                "indexed": upto,
                "snn_hit1": snn1 / n,
                "lex_hit1": lex1 / n,
                "snn_hit5": snn5 / n,
                "lex_hit5": lex5 / n,
            }
        )
        print(
            f"{day:3d} {upto:7d} {snn1 / n:10.4f} {lex1 / n:11.4f} {snn5 / n:10.4f} {lex5 / n:11.4f}",
            flush=True,
        )
    out = Path("sweep_results/lm_recall_snn/sim_stream.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"days": rows}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\n写入 {out}")
    print("注意：两臂都**没有在流上更新**——这是 U-1 的对照，不是 U-3。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
