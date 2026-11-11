"""U-3 的正式实验：**带漂移的流 + 一个能动的小头**。

设计要点（每一条都是为了不给自己放水）：

1. **流带漂移**：前一半记忆在第 1–4 天到达，后一半在第 5–8 天到达；
   每种渠道的相对强弱因此有机会随时间变；
2. **能动的只有小头**：3 路通道的权重。0.5B 主干从每天 10⁴ token 里动不起来
   （差五个数量级），这一点在 `design-inseparability.md` 里写死了；
3. **在线更新只用当天的查询、只在评测前用"昨天的权重"评**——否则就是拿答案调参；
4. 两臂：**冻结**（权重恒为均匀）vs **动态**（每天用当天查询做一次小扰动搜索）。

判据 U-3：动态臂在**未见过的**那些天上的 hit@1 高于冻结 **≥3pp**。

用法::

    uv run python scripts/sim_drift_u3.py --days 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sim_stream import _post, lexical_matrix

GOLD = Path("D:/Projects/BioSNN-MemoryTerminal/benchmarks/gold")
SNN_HOST = "http://127.0.0.1:18734"
OLLAMA = "http://127.0.0.1:11434"


def embed_snn(texts: list[str]) -> np.ndarray:
    out = _post(f"{SNN_HOST}/encode", {"texts": texts}, timeout=600)["vectors"]
    return np.asarray(out, dtype=np.float32)


def embed_nomic(texts: list[str], batch: int = 32) -> np.ndarray:
    rows = []
    for start in range(0, len(texts), batch):
        payload = {"model": "nomic-embed-text", "input": texts[start : start + batch]}
        rows.append(
            np.asarray(
                _post(f"{OLLAMA}/api/embed", payload, timeout=300)["embeddings"], dtype=np.float32
            )
        )
    matrix = np.concatenate(rows, axis=0)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-12)


def _rank(scores: np.ndarray, ids: list[str], want: set[str]) -> int | None:
    """名次（0 起）。**`rank` 是名次、`order[rank]` 才是下标**——这两者混过一次，报出过假数。"""
    order = np.argsort(-scores)
    return next((rank for rank, idx in enumerate(order.tolist()) if ids[idx] in want), None)


def score(ids, queries, vectors, weights) -> tuple[float, float]:
    """加权融合后的 (hit@1, MRR)。三路分数都在 [0,1] 量级上。"""
    hit1 = 0
    rr = 0.0
    for query, (s_lex, s_sem, s_snn) in zip(queries, vectors, strict=True):
        mixed = weights[0] * s_lex + weights[1] * s_sem + weights[2] * s_snn
        position = _rank(mixed, ids, set(query["expect"]))
        hit1 += int(position == 0)
        rr += (1.0 / (position + 1)) if position is not None else 0.0
    n = max(len(queries), 1)
    return hit1 / n, rr / n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="U-3：漂移流上的动态 vs 冻结")
    ap.add_argument("--days", type=int, default=8)
    ap.add_argument("--step", type=float, default=0.6, help="权重扰动步长")
    ap.add_argument("--gold", type=Path, default=GOLD)
    ap.add_argument(
        "--queries",
        type=Path,
        default=None,
        help="查询文件；默认 gold/queries.jsonl。换成改述集可把每天样本量提 5 倍",
    )
    args = ap.parse_args(argv)

    memories = [
        json.loads(x) for x in (args.gold / "memories.jsonl").read_text("utf-8").splitlines()
    ]
    query_path = args.queries or (args.gold / "queries.jsonl")
    queries = [json.loads(x) for x in query_path.read_text("utf-8").splitlines()]
    print(f"查询来源 {query_path.name}：{len(queries)} 条", flush=True)
    queries = [q for q in queries if q.get("expect")]
    ids = [m["id"] for m in memories]
    where = {mid: index for index, mid in enumerate(ids)}

    contents = [m["content"] for m in memories]
    texts = [q["query"] for q in queries]
    combined = contents + texts
    lex_all = lexical_matrix(combined)
    sem_all = embed_nomic(combined)
    snn_all = embed_snn(combined)
    lex_m, lex_q = lex_all[: len(contents)], lex_all[len(contents) :]
    sem_m, sem_q = sem_all[: len(contents)], sem_all[len(contents) :]
    snn_m, snn_q = snn_all[: len(contents)], snn_all[len(contents) :]
    print(
        f"三路就绪：词法 {lex_m.shape[1]} 维 / nomic {sem_m.shape[1]} / SNN {snn_m.shape[1]}",
        flush=True,
    )

    # 漂移：前一半先到，后一半后到
    half = len(memories) // 2
    order = list(range(half)) + list(range(half, len(memories)))
    per_day = max(1, len(memories) // args.days)
    arriving = [order[d * per_day : min((d + 1) * per_day, len(order))] for d in range(args.days)]

    # 每条查询归到"它的答案到达"的那一天
    buckets: list[list[int]] = [[] for _ in range(args.days + 1)]
    for index, query in enumerate(queries):
        first = min((where[mid] for mid in query["expect"] if mid in where), default=None)
        if first is None:
            continue
        day = next((d + 1 for d in range(args.days) if first in arriving[d]), args.days + 1)
        buckets[min(day, args.days)].append(index)

    w = np.ones(3) / 3.0
    frozen = np.ones(3) / 3.0
    rows = []
    seen: list[int] = []
    for day in range(1, args.days + 1):
        seen.extend(arriving[day - 1])
        keep = sorted(seen)
        table = {
            "lex": lex_m[keep],
            "sem": sem_m[keep],
            "snn": snn_m[keep],
        }
        keep_ids = [ids[i] for i in keep]
        rows_q = buckets[day]
        if rows_q:
            vecs = [
                (table["lex"] @ lex_q[i], table["sem"] @ sem_q[i], table["snn"] @ snn_q[i])
                for i in rows_q
            ]
            qs = [queries[i] for i in rows_q]
            f_hit, f_mrr = score(keep_ids, qs, vecs, frozen)
            d_hit, d_mrr = score(keep_ids, qs, vecs, w)
            rows.append(
                {
                    "day": day,
                    "queries": len(rows_q),
                    "frozen_hit1": f_hit,
                    "dynamic_hit1": d_hit,
                    "frozen_mrr": f_mrr,
                    "dynamic_mrr": d_mrr,
                    "weights": np.round(w, 3).tolist(),
                }
            )
            print(
                f"  天 {day}：{len(rows_q)} 条查询  冻结 {f_hit:.3f} / 动态 {d_hit:.3f}"
                f"  w={np.round(w, 2).tolist()}",
                flush=True,
            )
            # **更新只用当天的查询，且发生在评测之后**
            best, best_w = -1.0, w
            for axis in range(3):
                for delta in (-args.step, 0.0, args.step):
                    cand = w.copy()
                    cand[axis] = max(cand[axis] + delta, 0.05)
                    cand = cand / cand.sum()
                    hit, mrr = score(keep_ids, qs, vecs, cand)
                    key = hit + 0.1 * mrr
                    if key > best:
                        best, best_w = key, cand
            w = best_w
        else:
            print(f"  天 {day}：无查询", flush=True)

    if not rows:
        print("没有可用天数")
        return 0
    frozen_mean = float(np.mean([r["frozen_hit1"] for r in rows[1:]] or [0]))
    dynamic_mean = float(np.mean([r["dynamic_hit1"] for r in rows[1:]] or [0]))
    print(
        f"\n（从第 2 天起算，避开首日无历史）冻结 {frozen_mean:.4f} / 动态 {dynamic_mean:.4f}"
        f"  ⇒ **{dynamic_mean - frozen_mean:+.4f}**"
    )
    out = Path("sweep_results/lm_recall_snn/u3_drift.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"rows": rows, "frozen_mean": frozen_mean, "dynamic_mean": dynamic_mean},
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"写入 {out}")
    print("判据 U-3：动态 ≥ 冻结 +3pp。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
