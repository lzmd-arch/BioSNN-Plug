"""建**字符级**快半的语料：从 FineWeb-2 原文重新分词成字符 id 流。

## 为什么要有这一版

现在的快半用**千问词表的 32k 子集**——和慢半共用一套 token 化。这从构造上最大化了
两者的误差相关性，而我们的实测正是：F-0 = 3/52（千问错而快半对的位置极少）。

**要融合真的变强，得先让两半"错得不一样"。** 字符级是成本最低的换粒度方式：
- 不需要新语料（同一份 FineWeb-2）；
- 词表小（8k）⇒ 读出的 M 可以更宽而总参数不变；
- 错误结构与"子词"模型天然不同。

## 词表

按频次取前 `--vocab` 个字符，**不足的用字节回退**（UTF-8 逐字节）补齐常见字之外的一切。
和千问子集那版同样的纪律：**256 个单字节必须全在**，否则回退落不了地。

## 产出

`D:/AxoData/sel/lm/corpus/char_u16/`：
- `parts/*.u16` → 合并成 `corpus.u16`（uint16，字符 id）
- `chars.json`：id → 字符 的对照表（**必须随检查点一起存**，否则模型解不出字）
- `manifest.json`：字符数、id 数、每片明细
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SRC = Path("D:/AxoData/sel/lm/corpus/fineweb2_cmn/data")
OUT = Path("D:/AxoData/sel/lm/corpus/char_u16")


def iter_texts(path: Path):
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=["text"]):
        for text in batch.column("text").to_pylist():
            if text:
                yield text


def build_vocab(files: list[Path], size: int, sample_chars: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    total = 0
    for path in files:
        for text in iter_texts(path):
            counts.update(text)
            total += len(text)
            if total >= sample_chars:
                break
        if total >= sample_chars:
            break
    # **id 0–255 保留给字节**，字符从 256 起。这样任何 OOV 字符都能按 UTF-8 字节落地。
    keep = [ch for ch, _ in counts.most_common(max(size - 256, 1))]
    mapping = {ch: 256 + i for i, ch in enumerate(keep)}
    print(f"字符表 {len(mapping)} 个，样本 {total / 1e9:.2f}B 字", flush=True)
    return mapping


def encode_one(job: tuple[str, str, dict[str, int]]) -> tuple[str, int, int]:
    src, dst, mapping = job
    chars = encoded = 0
    with Path(dst).open("wb") as handle:
        for text in iter_texts(Path(src)):
            ids: list[int] = []
            for ch in text:
                slot = mapping.get(ch)
                if slot is not None:
                    ids.append(slot)
                else:
                    ids.extend(ch.encode("utf-8", "ignore"))  # 字节本身就是 id 0–255
            if ids:
                import numpy as np

                np.asarray(ids, dtype=np.uint16).tofile(handle)
            chars += len(text)
            encoded += len(ids)
    return Path(src).name, chars, encoded


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="建字符级语料")
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--vocab", type=int, default=8000)
    ap.add_argument("--sample-chars", type=int, default=200_000_000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 3))
    args = ap.parse_args(argv)

    files = sorted(args.src.glob("*.parquet"))
    if not files:
        raise SystemExit(f"没找到语料：{args.src}")
    args.out.mkdir(parents=True, exist_ok=True)

    mapping = build_vocab(files, args.vocab, args.sample_chars)
    (args.out / "chars.json").write_text(
        json.dumps({"chars": list(mapping)}, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    parts = args.out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), str(parts / (f.stem + ".u16")), mapping) for f in files]
    print(f"{len(files)} 片，{args.workers} 进程", flush=True)

    chars = encoded = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (name, c, e) in enumerate(pool.map(encode_one, jobs), 1):
            chars += c
            encoded += e
            print(f"  [{index}/{len(files)}] {name} 累计 {encoded / 1e9:.2f}B 字符 id", flush=True)

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
        "ids": encoded,
        "bin_bytes": target.stat().st_size,
        "vocab": len(mapping),
        "files": len(files),
        "note": "字符级；chars.json 必须随检查点一起存",
    }
    (args.out / "manifest.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
