"""语料分片的正文迭代器：**jsonl 与 parquet 都认**。

放在一个单独的小模块里，是因为有两处都要用（挑 32k 子集、建 id 流）。
复制一份的代价是两边迟早不一致——而这类不一致不会报错，只会让子集按 A 语料挑、
正文按 B 语料切。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

#: 目录里能被认作语料分片的扩展名。
SUFFIXES = (".jsonl", ".parquet")


def shards(directory: Path) -> list[Path]:
    """按文件名排序收集分片。顺序重要：同一时段的分布更一致。"""
    found: list[Path] = []
    for suffix in SUFFIXES:
        found.extend(directory.glob(f"*{suffix}"))
    return sorted(found)


def iter_texts(path: Path) -> Iterator[str]:
    """一个分片里的正文。"""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=["text"]):
            for text in batch.column("text").to_pylist():
                if text:
                    yield text
        return
    with path.open(encoding="utf-8", errors="ignore") as source:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text") or row.get("content") or ""
            if text:
                yield text
