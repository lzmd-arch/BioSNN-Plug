"""50k 时间窗：整行版本 vs 只取 id 版本。

时间窗召回只需要 id 来决定候选子集；整行版本会先把窗内每条记忆的正文与 SDR
解码一遍，随后 `_scan` 又按 id 取一次。这里量宽窗口（覆盖全库）上的差额。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import axo_substrate_probe as probe

AXO_SRC_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/src")
CORPUS_DEFAULT = Path(r"D:/Projects/BioSNN-MemoryTerminal/benchmarks/substrate")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axo-src", type=Path, default=AXO_SRC_DEFAULT)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DEFAULT)
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--out", type=Path, default=Path("sweep_results/axo_recall_50k/time_window.json"))
    args = parser.parse_args(argv)

    axo = probe.Axo(args.axo_src)
    from memcore.store import MemoryStore
    from memcore.types import MemoryItem, MemoryKind, Tier

    base = axo.load(args.corpus_dir)
    memories = axo.bench.scaled_memories(base.memories, args.n)
    contents = [memory.content for memory in memories]
    started = time.perf_counter()
    codes = [axo.sdr.encode_lexical(text) for text in contents]
    encode_s = time.perf_counter() - started

    store = MemoryStore(":memory:")
    for memory, content, code in zip(memories, contents, codes, strict=True):
        store.add(
            MemoryItem(
                memory_id=memory.memory_id,
                kind=MemoryKind.FACT,
                tier=Tier.RECENT,
                content=content,
                created_at=1_700_000_000.0,
                lex_sdr=code,
            )
        )

    since, until = 1_699_999_999.0, 1_700_000_001.0
    started = time.perf_counter()
    items = store.items_created_between(since, until)
    items_s = time.perf_counter() - started
    started = time.perf_counter()
    ids = store.ids_created_between(since, until)
    ids_s = time.perf_counter() - started
    assert ids == [item.memory_id for item in items]

    payload = {
        "n": args.n,
        "encode_s": encode_s,
        "window_items": len(items),
        "items_s": items_s,
        "ids_s": ids_s,
        "speedup": items_s / ids_s if ids_s else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(payload)
    print(f"[落盘] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
