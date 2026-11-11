"""增量分词：只编码 ``corpus.u16`` 还没吃进去的那些分片，产物是 ``corpus_b.u16``。

为什么单独写：``build_tokenized_corpus.py`` 每次都会**重训分词器并重编码全量**，
在 48 GB 上是几小时的重跑。这里复用已存的分词器（`bpe/vocab.json` + `bpe/merges.txt`），
只补新片，多进程并行。

产物与 `corpus.u16` **同词表**，可以直接接在一起 memmap。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DONE = [
    f"2020-40_zh_head_{i:04d}.jsonl" for i in range(8)
]  # 已进 corpus.u16 的 8 片（manifest 记的 files=8）


def _iter_texts(path: Path):
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text") or row.get("content") or ""
            if text:
                yield text


def encode_one(args: tuple[str, str, str]) -> tuple[str, int, int]:
    src, out, vocab_dir = args
    from tokenizers import ByteLevelBPETokenizer

    tokenizer = ByteLevelBPETokenizer(f"{vocab_dir}/vocab.json", f"{vocab_dir}/merges.txt")
    chars = tokens = 0
    with Path(out).open("wb") as handle:
        for text in _iter_texts(Path(src)):
            ids = tokenizer.encode(text).ids
            np.asarray(ids, dtype=np.uint16).tofile(handle)
            chars += len(text)
            tokens += len(ids)
    return Path(src).name, chars, tokens


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="增量分词：只补没吃过的分片")
    ap.add_argument("--src", type=Path, default=Path("D:/AxoData/sel/lm/corpus/skypile/data"))
    ap.add_argument("--tok", type=Path, default=Path("D:/AxoData/sel/lm/corpus/skypile_tok"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 4))
    args = ap.parse_args(argv)

    todo = [p for p in sorted(args.src.glob("*.jsonl")) if p.name not in DONE]
    if not todo:
        print("没有新分片")
        return 0
    print(f"待编码 {len(todo)} 片，{args.workers} 进程", flush=True)
    tmp = args.tok / "part"
    tmp.mkdir(parents=True, exist_ok=True)
    vocab_dir = str(args.tok / "bpe")
    jobs = [(str(p), str(tmp / (p.stem + ".u16")), vocab_dir) for p in todo]

    t0 = time.time()
    chars = tokens = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (name, c, t) in enumerate(pool.map(encode_one, jobs), 1):
            chars += c
            tokens += t
            rate = tokens / max(time.time() - t0, 1)
            print(
                f"  [{index}/{len(todo)}] {name} 累计 {tokens / 1e6:.0f}M token"
                f"（{rate / 1e3:.0f}k tok/s）",
                flush=True,
            )

    out = args.tok / "corpus_b.u16"
    with out.open("wb") as dst:
        for path in sorted(tmp.glob("*.u16")):
            with path.open("rb") as src:
                while chunk := src.read(1 << 24):
                    dst.write(chunk)
            path.unlink()
    tmp.rmdir()
    stats = {
        "chars": chars,
        "tokens": tokens,
        "chars_per_token": round(chars / max(tokens, 1), 3),
        "bin_bytes": out.stat().st_size,
        "files": len(todo),
    }
    (args.tok / "manifest_b.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
