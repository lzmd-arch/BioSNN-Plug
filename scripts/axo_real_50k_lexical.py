"""真实文本 50k：AISHELL-3 转写上的召回 profiling。

**这不是个人记忆库。** AISHELL-3 是公开中文语音语料，每行一条真实转写（中位 11 字）。
用它替代 substrate 的“同簇拼接放大”，量真实文本分布下的词法索引、候选与召回。
语料不复制进仓库，只登记路径、行数与哈希。

查询用**确定性的字符丢字**从库内条目派生（保留 ≥4 字），期望是原条目；
它量的是真实文本上的候选/召回鲁棒性，不是语义相关性金标准。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import axo_substrate_probe as probe

AXO_SRC_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/src")
AISHELL_DEFAULT = Path(r"D:/Projects/_scratch_ui/aishell3_full.jsonl")


def load_texts(path: Path, *, min_chars: int, n: int) -> tuple[list[str], str]:
    texts: list[str] = []
    seen: set[str] = set()
    digest = hashlib.sha256()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            digest.update(line.encode("utf-8"))
            row = json.loads(line)
            text = (row.get("text") or "").strip()
            if len(text) < min_chars or text in seen:
                continue
            seen.add(text)
            texts.append(text)
    if len(texts) < n:
        raise SystemExit(f"去重后只有 {len(texts)} 条，不够 {n}")
    rng = np.random.default_rng(0)
    picked = np.sort(rng.choice(len(texts), size=n, replace=False))
    return [texts[int(index)] for index in picked], digest.hexdigest()


def perturb(text: str, rng: np.random.Generator, *, keep: float = 0.7) -> str:
    chars = [char for char in text if not char.isspace()]
    count = max(4, int(round(len(chars) * keep)))
    keep_indices = np.sort(rng.choice(len(chars), size=min(count, len(chars)), replace=False))
    return "".join(chars[int(index)] for index in keep_indices)


def metrics(injected: list[str], expected: str):
    first = next((pos for pos, mid in enumerate(injected, start=1) if mid == expected), None)
    return (
        1.0 if injected and injected[0] == expected else 0.0,
        1.0 if first is not None and first <= 8 else 0.0,
        0.0 if first is None else 1.0 / first,
    )


def mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--aishell", type=Path, default=AISHELL_DEFAULT)
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--min-chars", type=int, default=6)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--full-scan-queries", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("sweep_results/axo_real_50k/lexical.json"))
    args = parser.parse_args(argv)

    axo = probe.Axo(args.axo_src)
    import memcore.recall as recall_module
    from memcore.recall import RecallConfig, recall
    from memcore.store import MemoryStore
    from memcore.types import MemoryItem, MemoryKind, Tier

    started = time.perf_counter()
    texts, source_digest = load_texts(args.aishell, min_chars=args.min_chars, n=args.n)
    load_s = time.perf_counter() - started

    started = time.perf_counter()
    codes = [axo.sdr.encode_lexical(text) for text in texts]
    encode_s = time.perf_counter() - started

    store = MemoryStore(":memory:")
    started = time.perf_counter()
    for index, (text, code) in enumerate(zip(texts, codes, strict=True)):
        store.add(
            MemoryItem(
                memory_id=f"a{index:06d}",
                kind=MemoryKind.FACT,
                tier=Tier.RECENT,
                content=text,
                created_at=1_700_000_000.0 + index,
                lex_sdr=code,
            )
        )
    add_s = time.perf_counter() - started
    started = time.perf_counter()
    store.lexical_index()
    index_s = time.perf_counter() - started

    rng = np.random.default_rng(1)
    query_indices = np.sort(rng.choice(args.n, size=args.queries, replace=False))
    queries = [
        (f"a{int(index):06d}", perturb(texts[int(index)], rng))
        for index in query_indices
    ]
    now = 1_700_000_000.0 + args.n + 1.0
    hit1, hit8, mrr, times, scanned = [], [], [], [], []
    for expected, query in queries:
        started = time.perf_counter()
        ledger = recall(
            store,
            query_text=query,
            budget_tokens=10**9,
            config=RecallConfig(),
            now=now,
            settle=False,
        )
        times.append(time.perf_counter() - started)
        scanned.append(ledger.scanned)
        one, eight, reciprocal = metrics(ledger.injected_ids, expected)
        hit1.append(one)
        hit8.append(eight)
        mrr.append(reciprocal)

    capped = [(expected, query) for expected, query in queries[: args.full_scan_queries]]
    original_prefilter = recall_module._prefilter_ids
    full = []
    try:
        recall_module._prefilter_ids = lambda *a, **k: None
        for expected, query in capped:
            started = time.perf_counter()
            ledger = recall(
                store,
                query_text=query,
                budget_tokens=10**9,
                config=RecallConfig(),
                now=now,
                settle=False,
            )
            full.append(
                (
                    tuple(ledger.injected_ids[:8]),
                    time.perf_counter() - started,
                    ledger.scanned,
                    ledger.injected_ids,
                )
            )
    finally:
        recall_module._prefilter_ids = original_prefilter

    candidate = []
    for expected, query in capped:
        ledger = recall(
            store,
            query_text=query,
            budget_tokens=10**9,
            config=RecallConfig(),
            now=now,
            settle=False,
        )
        candidate.append(tuple(ledger.injected_ids[:8]))
    top1_same = mean(
        [1.0 if a[:1] == b[0][:1] else 0.0 for a, b in zip(candidate, full, strict=True)]
    )
    top8_same = mean(
        [1.0 if a[:8] == b[0] else 0.0 for a, b in zip(candidate, full, strict=True)]
    )

    payload = {
        "source": str(args.aishell),
        "source_sha256": source_digest,
        "n": args.n,
        "min_chars": args.min_chars,
        "load_s": load_s,
        "encode_s": encode_s,
        "add_s": add_s,
        "index_s": index_s,
        "queries": len(queries),
        "hit1": mean(hit1),
        "hit8": mean(hit8),
        "mrr": mean(mrr),
        "recall_ms": float(np.mean(times) * 1000),
        "scanned": float(np.mean(scanned)),
        "full_scan_queries": len(full),
        "full_scan_ms": float(np.mean([item[1] for item in full]) * 1000),
        "full_scan_scanned": float(np.mean([item[2] for item in full])),
        "top1_same": top1_same,
        "top8_same": top8_same,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(payload)
    print(f"[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
