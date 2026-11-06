"""量 R1 三档合成语料的 bigram（判据 R1-0 / R1-1 的前置检查）。

按生成时写死的六折 `group` 切分，每折独立建词表、独立拟合 bigram。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = Path("D:/AxoData/sel/lm/v1/synth")
OUT = REPO / "sweep_results" / "lm_eprop_r1"
LEVELS = ("E", "M", "H")
FOLDS = 6
BIGRAM_K = 0.5
RANGE_MIN = 0.10
FLOOR_MIN = 0.15


def load(level: str) -> dict[int, list[str]]:
    groups: dict[int, list[str]] = defaultdict(list)
    for line in (DATA / f"{level}.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        groups[int(row["group"])].append(f"\x01{row['user']}\x02{row['assistant']}\x03")
    return dict(groups)


def bigram(train: list[str], test: list[str]) -> tuple[float, float]:
    vocab = sorted({ch for text in train for ch in text})
    index = {ch: i for i, ch in enumerate(vocab)}
    unk = len(vocab)
    encode = lambda text: [index.get(ch, unk) for ch in text]  # noqa: E731
    counts: dict[int, Counter] = defaultdict(Counter)
    for text in train:
        for left, right in pairwise(encode(text)):
            counts[left][right] += 1
    size = len(vocab) + 1
    hits = total = 0
    ce = 0.0
    for text in test:
        for left, right in pairwise(encode(text)):
            row = counts.get(left)
            if not row:
                total += 1
                ce += -math.log(1.0 / size)
                continue
            denominator = sum(row.values()) + BIGRAM_K * size
            hits += int(max(row, key=row.get) == right)
            ce += -math.log((row.get(right, 0) + BIGRAM_K) / denominator)
            total += 1
    return hits / max(total, 1), ce / max(total, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="量三档合成语料的 bigram")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="ladder_bigram")
    args = parser.parse_args(argv)

    report: dict[str, dict] = {}
    for level in LEVELS:
        groups = load(level)
        accs, ces = [], []
        for held in sorted(groups):
            train = [text for g in groups if g != held for text in groups[g]]
            test = groups[held]
            acc, ce = bigram(train, test)
            accs.append(acc)
            ces.append(ce)
        chars = sum(len(text) for texts in groups.values() for text in texts)
        chars_per_pair = chars / max(sum(len(v) for v in groups.values()), 1)
        report[level] = {
            "pairs": sum(len(v) for v in groups.values()),
            "chars": chars,
            "chars_per_pair": round(chars_per_pair, 1),
            "unique_chars": len({ch for texts in groups.values() for t in texts for ch in t}),
            "fold_top1": [round(a, 4) for a in accs],
            "fold_ce": [round(c, 3) for c in ces],
            "bigram_top1": round(statistics.mean(accs), 4),
            "bigram_ce": round(statistics.mean(ces), 3),
        }
        print(f"[{level}] {report[level]}", flush=True)

    values = [report[level]["bigram_top1"] for level in LEVELS]
    spread = max(values) - min(values)
    criteria = {
        "r1_0_spread_ok": spread >= RANGE_MIN,
        "r1_1_floor_ok": min(values) >= FLOOR_MIN,
        "spread": round(spread, 4),
        "min_bigram": round(min(values), 4),
        "ordered": values == sorted(values, reverse=True),
    }
    out = {"levels": report, "criteria": criteria, "folds": FOLDS, "k": BIGRAM_K}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"三档 bigram：E {values[0]:.3f} / M {values[1]:.3f} / H {values[2]:.3f}；极差 {spread:.3f}"
    )
    flags = " ".join(
        f"{k}={'过' if v else '不过'}" for k, v in criteria.items() if isinstance(v, bool)
    )
    print(f"[判据] {flags}")
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
