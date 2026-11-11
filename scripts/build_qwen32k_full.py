"""把**全量语料**编成千问 32k 子集的 id 流（uint16），多进程按片并行。

与 `build_qwen_subset_corpus.py` 的区别只有一条：那个是单进程、只做样本；
这个是全量、按片并行，产物可以直接喂给 `run_1b`。

存储用**重映射后的本地 id**（千问 id 最大 151,935，装不进 uint16）。
映射表 `id_map.json` 与 `subset_32k.json` 一起构成"我们的 id ↔ 千问 id"的桥。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

QWEN = Path("D:/AxoData/sel/lm/corpus/qwen_tok")
CORPUS = Path("D:/AxoData/sel/lm/corpus/skypile/data")
OUT = Path("D:/AxoData/sel/lm/corpus/qwen32k_full")


def bytes_to_unicode() -> dict[int, str]:
    keep = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = keep[:]
    extra = 0
    for byte in range(256):
        if byte not in keep:
            keep.append(byte)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(keep, [chr(c) for c in chars], strict=True))


def encode_one(job: tuple[str, str, str]) -> tuple[str, int, int]:
    src, dst, qwen_dir = job
    from tokenizers import ByteLevelBPETokenizer

    root = Path(qwen_dir)
    subset = json.loads((root / "subset_32k.json").read_text(encoding="utf-8"))["ids"]
    remap = {qwen_id: local for local, qwen_id in enumerate(subset)}
    tokenizer = ByteLevelBPETokenizer(str(root / "vocab.json"), str(root / "merges.txt"))
    vocab = json.loads((root / "vocab.json").read_text(encoding="utf-8"))
    to_byte = bytes_to_unicode()
    byte_local = [remap[vocab[to_byte[b]]] for b in range(256)]

    chars = qwen_tokens = 0
    with (
        Path(dst).open("wb") as handle,
        Path(src).open(encoding="utf-8", errors="ignore") as source,
    ):
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text") or row.get("content") or ""
            if not text:
                continue
            ids = tokenizer.encode(text).ids
            local: list[int] = []
            for qwen_id in ids:
                slot = remap.get(qwen_id)
                if slot is None:
                    piece = tokenizer.decode([qwen_id]).encode("utf-8", "ignore")
                    local.extend(byte_local[b] for b in piece)
                else:
                    local.append(slot)
            np.asarray(local, dtype=np.uint16).tofile(handle)
            chars += len(text)
            qwen_tokens += len(ids)
    return Path(src).name, chars, qwen_tokens


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="全量语料 -> 千问 32k 子集 id 流（并行）")
    ap.add_argument("--qwen", type=Path, default=QWEN)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 3))
    args = ap.parse_args(argv)

    files = sorted(args.corpus.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"没找到语料：{args.corpus}")
    parts = args.out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), str(parts / (f.stem + ".u16")), str(args.qwen)) for f in files]
    print(f"{len(files)} 片，{args.workers} 进程", flush=True)

    started = time.time()
    chars = tokens = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (name, c, t) in enumerate(pool.map(encode_one, jobs), 1):
            chars += c
            tokens += t
            rate = tokens / max(time.time() - started, 1)
            print(
                f"  [{index}/{len(files)}] {name} 累计 {tokens / 1e9:.2f}B token"
                f"（{rate / 1e3:.0f}k tok/s）",
                flush=True,
            )

    target = args.out / "corpus.u16"
    with target.open("wb") as dst:
        for part in sorted(parts.glob("*.u16")):
            with part.open("rb") as src:
                while chunk := src.read(1 << 24):
                    dst.write(chunk)
            part.unlink()
    parts.rmdir()
    stats = {
        "chars": chars,
        "qwen_tokens": tokens,
        "bin_bytes": target.stat().st_size,
        "files": len(files),
        "subset_size": 32000,
        "note": "本地 id 是 subset_32k.json 里的下标；融合时用 id_map 换回千问 id",
    }
    (args.out / "manifest.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
