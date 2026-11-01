"""512 个 ALIF 神经元：召回（A）与门控（B）两处的最小可行试验。

判据见同目录 `prereg.md`（跑之前写死）。这里只负责把两种读法跑出来，不做调参。

    uv run python research/spiking_512/run_512.py --seeds 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import zlib
from pathlib import Path

import numpy as np
from biosnn_bus import AdaptiveMembrane

WIDTH, K, N_NEURONS, T_STEPS = 4096, 64, 512, 20


def assoc_code(text: str) -> np.ndarray:
    r"""k-of-n 稀疏码：**字符二元组**取位再合并。

    两次修正都记在这里，因为它们改变了结论：
    ① 整句哈希 → 共享词不重叠（基线 hit@1 1/85）；
    ② `re.split(r"\W+")` 对中文无效（整句一个 token），改成字符二元组后才有词法结构。
    """
    clean = "".join(ch for ch in text.lower() if not ch.isspace())
    grams = [clean[i : i + 2] for i in range(len(clean) - 1)] or [clean]
    per_gram = max(1, K // len(grams))
    bits: set[int] = set()
    for gram in grams:
        rng = np.random.default_rng(zlib.crc32(gram.encode("utf-8")))
        bits.update(int(bit) for bit in rng.choice(WIDTH, size=per_gram, replace=False))
    return np.sort(np.array(sorted(bits)[:K], dtype=int))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return len(np.intersect1d(a, b)) / K


class Population:
    """512 个 ALIF 神经元 + 固定的 4096→512 投影。"""

    def __init__(self, seed: int, gain: float = 1.0) -> None:
        rng = np.random.default_rng(seed)
        self.proj = rng.normal(0.0, 1.0 / np.sqrt(K), size=(WIDTH, N_NEURONS))
        self.membrane = AdaptiveMembrane(N_NEURONS)
        self.gain = gain

    def current(self, *codes: np.ndarray) -> np.ndarray:
        idx = np.concatenate(codes)
        return self.proj[idx].sum(axis=0) * self.gain

    def spikes(self, *codes: np.ndarray) -> int:
        self.membrane.reset()
        current = self.current(*codes)
        return int(sum(int(self.membrane.forward(current).sum()) for _ in range(T_STEPS)))


def load_gold(root: Path) -> tuple[dict[str, str], list[dict]]:
    memories = {
        row["id"]: row["content"]
        for row in map(json.loads, (root / "memories.jsonl").read_text("utf-8").splitlines())
    }
    queries = list(map(json.loads, (root / "queries.jsonl").read_text("utf-8").splitlines()))
    return memories, queries


def run(seed: int, memories: dict[str, str], queries: list[dict], prefilter: int) -> dict:
    pop = Population(seed)
    codes = {mid: assoc_code(text) for mid, text in memories.items()}

    hit1 = hit5 = base1 = base5 = 0
    feats, labels = [], []
    for query in queries:
        q_code = assoc_code(query["query"])
        expect = set(query.get("expect") or [])
        ranked = sorted(codes, key=lambda mid: cosine(q_code, codes[mid]), reverse=True)[:prefilter]

        base_hits = [mid for mid in ranked if mid in expect]
        base1 += bool(base_hits and base_hits[0] == ranked[0])
        base5 += bool(base_hits and base_hits[0] in ranked[:5])

        scored = []
        for mid in ranked:
            both = pop.spikes(q_code, codes[mid])
            alone = max(pop.spikes(q_code), pop.spikes(codes[mid]))
            scored.append((both - alone, mid))
        scored.sort(reverse=True)
        top = [mid for _, mid in scored]
        hit1 += bool(expect and top[0] in expect)
        hit5 += bool(expect and any(mid in expect for mid in top[:5]))

        feats.append(pop.spikes(q_code) / (N_NEURONS * T_STEPS))
        labels.append(1 if expect else 0)

    return {
        "seed": seed,
        "spiking_hit1": hit1 / len(queries),
        "spiking_hit5": hit5 / len(queries),
        "baseline_hit1": base1 / len(queries),
        "baseline_hit5": base5 / len(queries),
        "feature": feats,
        "label": labels,
    }


def auroc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold", type=Path, default=Path("D:/Projects/BioSNN-MemoryTerminal/benchmarks/gold")
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--prefilter", type=int, default=50)
    args = parser.parse_args(argv)

    memories, queries = load_gold(args.gold)
    rows = [run(seed, memories, queries, args.prefilter) for seed in range(1, args.seeds + 1)]

    def stat(key: str) -> tuple[float, float, float]:
        values = sorted(row[key] for row in rows)
        return statistics.median(values), values[0], values[-1]

    s1, b1 = stat("spiking_hit1"), stat("baseline_hit1")
    s5, b5 = stat("spiking_hit5"), stat("baseline_hit5")
    print(
        f"gold：{len(memories)} 条记忆 / {len(queries)} 条查询；种子 {len(rows)}；预筛前 {args.prefilter} 条"
    )
    print(
        f"A1 hit@1  群体 {s1[0]:.3f} [{s1[1]:.3f}, {s1[2]:.3f}]   基线 {b1[0]:.3f} [{b1[1]:.3f}, {b1[2]:.3f}]   差 {s1[0] - b1[0]:+.3f}"
    )
    print(f"A2 hit@5  群体 {s5[0]:.3f}   基线 {b5[0]:.3f}   差 {s5[0] - b5[0]:+.3f}")
    aucs = [auroc(row["feature"], row["label"]) for row in rows]
    print(f"B  AUROC  群体 {statistics.median(aucs):.3f} [{min(aucs):.3f}, {max(aucs):.3f}]")
    print(f"A1 判据（≥ +0.05）：{'过' if s1[0] - b1[0] >= 0.05 else '不过'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
