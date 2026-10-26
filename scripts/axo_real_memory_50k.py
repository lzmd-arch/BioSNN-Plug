"""真实记忆库达到阈值时跑一次 50k 召回基准。

**只读**：用 SQLite 在线备份把 ``D:/AxoData/memory/mem.db`` 复制到临时目录，原库
不打开写。输出只含聚合指标，不写正文、查询或记忆 id。

触发方式见 ``run_real_50k_trigger.ps1``：每日检查；库未到 ``--threshold`` 时直接
退出，到了且没有既有结果时才跑。``--force`` 可手动重跑。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import axo_real_50k_lexical as real
import axo_substrate_probe as probe

AXO_SRC_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/src")
DATA_ROOT_DEFAULT = Path(r"D:/AxoData")
OUT_DEFAULT = Path(r"D:/Projects/BioSNN-Plug-neuron/sweep_results/axo_real_50k/real_memory.json")


def copy_db(source_path: Path) -> tuple[Path, Path]:
    directory = Path(tempfile.mkdtemp(prefix="axo_real_50k_"))
    target = directory / "mem.db"
    source = sqlite3.connect(f"file:{source_path.as_posix()}?mode=ro", uri=True)
    destination = sqlite3.connect(target)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    return directory, target


def pick_space(store):
    spaces = store.registry.all()
    for space in spaces:
        if space.dim == 768 and "nomic" in space.model_ref.lower():
            return space
    return spaces[0] if spaces else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--threshold", type=int, default=50000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--full-scan-queries", type=int, default=10)
    parser.add_argument("--min-chars", type=int, default=6)
    parser.add_argument("--no-semantic", action="store_true")
    parser.add_argument(
        "--require-semantic",
        action="store_true",
        help="语义通道不可用时退出 2，不写最终结果（计划任务用，次日重试）",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args(argv)
    if args.require_semantic and args.no_semantic:
        parser.error("--require-semantic 与 --no-semantic 不能同时给")

    source = args.data_root / "memory" / "mem.db"
    if not source.is_file():
        print(f"not_ready: 找不到真实库 {source}")
        return 0
    if args.out.exists() and not args.force:
        try:
            previous = json.loads(args.out.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        if int(previous.get("count", 0)) >= args.threshold:
            print(f"already_done: {args.out} 已有 count={previous.get('count')}")
            return 0

    axo = probe.Axo(args.axo_src)
    import memcore.recall as recall_module
    from memcore.recall import RecallConfig, recall
    from memcore.store import MemoryStore

    temporary, copied = copy_db(source)
    try:
        store = MemoryStore(str(copied))
        count = store.count()
        print(f"count={count} threshold={args.threshold}")
        if count < args.threshold:
            print(f"not_ready: {count} < {args.threshold}")
            return 0

        started = time.perf_counter()
        store.lexical_index()
        index_s = time.perf_counter() - started
        items = [item for item in store.iter_items() if len(item.content.strip()) >= args.min_chars]
        if len(items) < args.queries:
            print(f"not_ready: 可用条目只有 {len(items)} 条")
            return 0

        rng = np.random.default_rng(1)
        picked = np.sort(rng.choice(len(items), size=args.queries, replace=False))
        queries = [
            (items[int(index)].memory_id, real.perturb(items[int(index)].content, rng))
            for index in picked
        ]
        now = max(item.created_at for item in items) + 1.0

        def run(selected, vectors=None, space_id=""):
            hit1, hit8, mrr, times, scanned, injected = [], [], [], [], [], []
            pairs = list(
                zip(
                    selected,
                    vectors if vectors is not None else [None] * len(selected),
                    strict=True,
                )
            )
            for (expected, query), vector in pairs:
                started = time.perf_counter()
                ledger = recall(
                    store,
                    query_text=query,
                    query_vector=vector,
                    space_id=space_id,
                    budget_tokens=10**9,
                    config=RecallConfig(),
                    now=now,
                    settle=False,
                )
                times.append(time.perf_counter() - started)
                scanned.append(ledger.scanned)
                injected.append(tuple(ledger.injected_ids[:8]))
                one, eight, reciprocal = real.metrics(ledger.injected_ids, expected)
                hit1.append(one)
                hit8.append(eight)
                mrr.append(reciprocal)
            return {
                "queries": len(selected),
                "hit1": real.mean(hit1),
                "hit8": real.mean(hit8),
                "mrr": real.mean(mrr),
                "recall_ms": float(np.mean(times) * 1000),
                "scanned": float(np.mean(scanned)),
            }, injected

        full_n = min(args.full_scan_queries, len(queries))
        full_selected = queries[:full_n]

        def full_scan(vectors, space_id):
            original = recall_module._prefilter_ids
            recall_module._prefilter_ids = lambda *a, **k: None
            try:
                return run(full_selected, vectors=vectors, space_id=space_id)
            finally:
                recall_module._prefilter_ids = original

        result = {
            "source": str(source),
            "count": count,
            "threshold": args.threshold,
            "min_chars": args.min_chars,
            "index_s": index_s,
            "full_scan_queries": full_n,
        }
        lexical, lexical_injected = run(queries)
        lexical_full, lexical_full_injected = full_scan(None, "")
        lexical["full_scan"] = lexical_full
        lexical["top1_same"] = real.mean(
            [
                1.0 if a[:1] == b[:1] else 0.0
                for a, b in zip(lexical_injected[:full_n], lexical_full_injected, strict=True)
            ]
        )
        lexical["top8_same"] = real.mean(
            [
                1.0 if a[:8] == b[:8] else 0.0
                for a, b in zip(lexical_injected[:full_n], lexical_full_injected, strict=True)
            ]
        )
        result["lexical"] = lexical

        space = pick_space(store)
        if args.require_semantic and space is None:
            print("semantic_unavailable: 库里没有可用的 768 维 nomic 空间")
            return 2
        if not args.no_semantic and space is not None:
            try:
                from axo.llm.ollama import OllamaEmbeddingBackend

                model = space.model_ref.split("/")[-1]
                backend = OllamaEmbeddingBackend(model=model, dim=space.dim)
                vectors = backend.embed([query for _, query in queries])
                semantic, semantic_injected = run(queries, vectors=vectors, space_id=space.space_id)
                semantic_full, semantic_full_injected = full_scan(vectors[:full_n], space.space_id)
                semantic["full_scan"] = semantic_full
                semantic["top1_same"] = real.mean(
                    [
                        1.0 if a[:1] == b[:1] else 0.0
                        for a, b in zip(
                            semantic_injected[:full_n], semantic_full_injected, strict=True
                        )
                    ]
                )
                semantic["top8_same"] = real.mean(
                    [
                        1.0 if a[:8] == b[:8] else 0.0
                        for a, b in zip(
                            semantic_injected[:full_n], semantic_full_injected, strict=True
                        )
                    ]
                )
                result["semantic"] = semantic
                result["space"] = {
                    "space_id": space.space_id,
                    "dim": space.dim,
                    "model_ref": space.model_ref,
                    "embeddings": store.count_embeddings(space.space_id),
                }
            except Exception as exc:  # noqa: BLE001 — 语义不可用不该让触发失败
                if args.require_semantic:
                    print(f"semantic_unavailable: {type(exc).__name__}: {exc}")
                    return 2
                result["semantic_error"] = f"{type(exc).__name__}: {exc}"

        result["completed_at"] = time.time()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False))
        print(f"[落盘] {args.out}")
        return 0
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
