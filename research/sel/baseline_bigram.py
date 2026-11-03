"""字符级 bigram 基线：给 P4-1 一个分母。

加 k 平滑；`<bos>` 开头、`<eos>` 结尾，按教师 forcing 逐位置算下一字符 top-1。
不做任何神经训练——它存在的意义是"学生的 top-1 必须比它高 10pp"，而不是它自己好看。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"


def load_pairs(path: Path) -> dict[str, list[str]]:
    splits: dict[str, list[str]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        splits[row["split"]].append(row["text"])
    return dict(splits)


def build_vocab(texts: list[str]) -> list[str]:
    chars = sorted({ch for text in texts for ch in text})
    return [BOS, EOS, UNK, *chars]


def encode(text: str, index: dict[str, int]) -> list[int]:
    unk = index[UNK]
    return [index[BOS], *(index.get(ch, unk) for ch in text), index[EOS]]


def train_counts(sequences: list[list[int]], size: int) -> list[list[float]]:
    counts = [[0.0] * size for _ in range(size)]
    for seq in sequences:
        for left, right in pairwise(seq):
            counts[left][right] += 1.0
    return counts


def top1_accuracy(counts: list[list[float]], sequences: list[list[int]], k: float) -> float:
    size = len(counts)
    hits = total = 0
    for seq in sequences:
        for left, right in pairwise(seq):
            row = [
                (counts[left][j] + k) if j != left else (counts[left][j] + k) for j in range(size)
            ]
            best = max(range(size), key=row.__getitem__)
            hits += int(best == right)
            total += 1
    return hits / max(1, total)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--k", type=float, default=0.1)
    args = parser.parse_args()
    splits = load_pairs(args.data)
    vocab = build_vocab(splits["train"])
    index = {ch: i for i, ch in enumerate(vocab)}
    train = [encode(text, index) for text in splits["train"]]
    counts = train_counts(train, len(vocab))
    report = {"vocab": len(vocab), "train": len(train), "k": args.k}
    for split in ("valid", "test"):
        seqs = [encode(text, index) for text in splits.get(split, [])]
        report[split] = round(top1_accuracy(counts, seqs, args.k), 4)
        report[f"{split}_n"] = len(seqs)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
