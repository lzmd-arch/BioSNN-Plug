"""把 JSONL 语料**预先分词成 uint16 二进制**，训练时用 memmap 随机取窗。

为什么要这一步：语料从 1.5M 字涨到 2.8B 字（1800 倍）。若训练时每批现解析 JSON、现切分，
**CPU 会先于 GPU 饱和** —— 2 天的预算会被数据管线吃掉，而不是被模型吃掉。

产物：单个 ``.bin``（uint16，V<65536）+ 一个 ``.json`` 记账。
训练侧用 ``numpy.memmap(..., dtype=np.uint16)`` 直接切片，热路径上没有 Python。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.sel.lm.bpe import train_tokenizer


def iter_texts(paths: list[Path]):
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row.get("text") or row.get("content") or ""
                if text:
                    yield text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="语料 -> uint16 分词二进制")
    ap.add_argument("--src", type=Path, default=Path("D:/AxoData/sel/lm/corpus/skypile/data"))
    ap.add_argument("--out", type=Path, default=Path("D:/AxoData/sel/lm/corpus/skypile_tok"))
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument(
        "--train-chars",
        type=int,
        default=200_000_000,
        help="学 BPE 合并用的字符数（分词器只需样本，不必全量）",
    )
    ap.add_argument("--limit-chars", type=int, default=0, help="0 = 全量")
    args = ap.parse_args(argv)

    files = sorted(args.src.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"没找到语料：{args.src}")
    print(f"语料 {len(files)} 个文件：{args.src}", flush=True)

    # 1) 学分词器（Rust，在样本上）
    sample_dir = args.out / "bpe_sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample = sample_dir / "sample.txt"
    taken = 0
    with sample.open("w", encoding="utf-8", newline="\n") as handle:
        for text in iter_texts(files):
            handle.write(text.replace("\n", " ") + chr(10))
            taken += len(text)
            if taken >= args.train_chars:
                break
    print(f"分词器样本 {taken:,} 字", flush=True)
    t0 = time.time()
    tokenizer = train_tokenizer([sample], args.out / "bpe", vocab_size=args.vocab)
    print(f"BPE 训练完成（{time.time() - t0:.0f}s，V={args.vocab}）", flush=True)

    # 2) 全量编码成 uint16
    args.out.mkdir(parents=True, exist_ok=True)
    bin_path = args.out / "corpus.u16"
    chars = tokens = 0
    started = time.time()
    with bin_path.open("wb") as handle:
        for text in iter_texts(files):
            ids = tokenizer.encode(text).ids
            np.asarray(ids, dtype=np.uint16).tofile(handle)
            chars += len(text)
            tokens += len(ids)
            if tokens % 50_000_000 < len(ids):
                rate = tokens / max(time.time() - started, 1)
                print(
                    f"  {tokens / 1e6:.0f}M token（{chars / 1e9:.2f}B 字），{rate / 1e3:.0f}k tok/s",
                    flush=True,
                )
            if args.limit_chars and chars >= args.limit_chars:
                break
    stats = {
        "chars": chars,
        "tokens": tokens,
        "tokens_per_char": round(tokens / max(chars, 1), 4),
        "chars_per_token": round(chars / max(tokens, 1), 3),
        "bin_bytes": bin_path.stat().st_size,
        "vocab": args.vocab,
        "files": len(files),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1) + chr(10), encoding="utf-8", newline="\n"
    )
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
