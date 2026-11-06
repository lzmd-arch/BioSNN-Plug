"""R0：真实会话语料上的字符级 next-token（判据见 `prereg-lm-r0-real-corpus.md`）。

留一会话（六折）；每折独立建词表、独立训练、独立评测；基线是同折的字符 bigram。
**数据不出本机**：只读 `D:/AxoData/sel/lm/v1/real.jsonl`，结果只写聚合指标。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

import torch

from research.sel.lm.model import TokenEProp
from research.sel.lm.run_m0 import BATCH, FAN_IN, N_REC, make_batches

REPO = Path(__file__).resolve().parents[3]
DATA = Path("D:/AxoData/sel/lm/v1/real.jsonl")
OUT = REPO / "sweep_results" / "lm_eprop_r0"

UPDATES = 2000
MAX_LEN = 192
TAU = 20.0
TAU_READ = 8.0
INPUT_STD = 0.12
LR_IN = 1e-3
LR_OUT = 0.5
SEED = 0
BIGRAM_K = 0.5

#: 判据（改这里必须改预注册）。
LOSS_DROP = 0.6
WIN_DELTA = 0.03
WIN_FOLDS = 4


def load_sequences(path: Path) -> dict[str, list[str]]:
    by_session: dict[str, list[str]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        by_session[row["session"]].append(f"\x01{row['user']}\x02{row['assistant']}\x03")
    return dict(by_session)


def build_vocab(texts: list[str]) -> dict[str, int]:
    chars = sorted({ch for text in texts for ch in text})
    return {token: index for index, token in enumerate(["<unk>", *chars])}


def encode(text: str, vocab: dict[str, int], *, cap: int) -> list[int]:
    unk = vocab["<unk>"]
    return [vocab.get(ch, unk) for ch in text][:cap]


def bigram_accuracy(
    train: list[list[int]], test: list[list[int]], size: int
) -> tuple[float, float]:
    counts: dict[int, Counter] = defaultdict(Counter)
    for sequence in train:
        for left, right in pairwise(sequence):
            counts[left][right] += 1
    hits = total = 0
    ce = 0.0
    for sequence in test:
        for left, right in pairwise(sequence):
            row = counts.get(left)
            if not row:
                total += 1
                ce += -math.log(1.0 / max(size, 2))
                continue
            denominator = sum(row.values()) + BIGRAM_K * size
            hits += int(max(row, key=row.get) == right)
            ce += -math.log((row.get(right, 0) + BIGRAM_K) / denominator)
            total += 1
    return hits / max(total, 1), ce / max(total, 1)


def run_fold(train_texts: list[str], test_texts: list[str], device, updates: int) -> dict:
    torch.manual_seed(SEED)
    vocab = build_vocab(train_texts)
    train = [encode(text, vocab, cap=MAX_LEN) for text in train_texts]
    test = [encode(text, vocab, cap=MAX_LEN) for text in test_texts]
    pad = vocab["<unk>"]
    model = TokenEProp(
        len(vocab),
        N_REC,
        fan_in=FAN_IN,
        tau=TAU,
        tau_read=TAU_READ,
        input_std=INPUT_STD,
        lr_in=LR_IN,
        lr_out=LR_OUT,
        recurrence="none",
        generator=torch.Generator().manual_seed(SEED),
        device=device,
    )
    generator = torch.Generator().manual_seed(SEED + 17)
    losses: list[float] = []
    for _ in range(updates):
        order = torch.randperm(len(train), generator=generator)[:BATCH].tolist()
        tokens, valid = make_batches(train, order=order, pad=pad)
        result = model.run(
            tokens.to(device), model.new_trace(tokens.shape[0]), valid=valid.to(device)
        )
        losses.append(result["ce"])
    tokens, valid = make_batches(test, order=list(range(len(test))), pad=pad)
    metrics = model.evaluate(
        tokens.to(device), model.new_trace(tokens.shape[0]), valid=valid.to(device)
    )
    base_top1, base_ce = bigram_accuracy(train, test, len(vocab))
    head = statistics.median(losses[:200])
    tail = statistics.median(losses[-200:])
    return {
        "vocab": len(vocab),
        "train_pairs": len(train),
        "test_pairs": len(test),
        "top1": round(metrics["top1"], 4),
        "ce": round(metrics["ce"], 4),
        "bigram_top1": round(base_top1, 4),
        "bigram_ce": round(base_ce, 4),
        "delta_top1": round(metrics["top1"] - base_top1, 4),
        "ce_ratio": round(tail / head, 4) if head else None,
        "tokens": metrics["tokens"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R0：真语料")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--tag", default="r0")
    args = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    by_session = load_sequences(args.data)
    sessions = sorted(by_session)
    print(f"会话 {len(sessions)} 个，各折对数 {[len(by_session[s]) for s in sessions]}", flush=True)

    folds: dict[str, dict] = {}
    for held in sessions:
        train_texts = [text for s in sessions if s != held for text in by_session[s]]
        folds[held] = run_fold(train_texts, by_session[held], device, args.updates)
        print(f"[hold-out {held}] {folds[held]}", flush=True)

    deltas = [fold["delta_top1"] for fold in folds.values()]
    ce_deltas = [fold["bigram_ce"] - fold["ce"] for fold in folds.values()]
    mean_delta = statistics.mean(deltas)
    criteria = {
        "r0_0_loss_falls": all(
            fold["ce_ratio"] is not None and fold["ce_ratio"] < LOSS_DROP for fold in folds.values()
        ),
        "r0_1_beats_bigram": mean_delta >= WIN_DELTA,
        "r0_2_most_folds_win": sum(1 for delta in deltas if delta > 0) >= WIN_FOLDS,
        "r0_3_ce_better": statistics.mean(ce_deltas) > 0,
        "mean_delta_top1": round(mean_delta, 4),
        "folds_winning": sum(1 for delta in deltas if delta > 0),
        "mean_top1": round(statistics.mean(f["top1"] for f in folds.values()), 4),
        "mean_bigram_top1": round(statistics.mean(f["bigram_top1"] for f in folds.values()), 4),
    }
    out = {
        "kind": "R0: feedforward local-rule LM on a real conversation corpus",
        "seed": SEED,
        "config": {
            "n_rec": N_REC,
            "fan_in": FAN_IN,
            "max_len": MAX_LEN,
            "batch": BATCH,
            "updates": args.updates,
            "lr_in": LR_IN,
            "lr_out": LR_OUT,
        },
        "folds": folds,
        "criteria": criteria,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"top1 均值 {criteria['mean_top1']:.4f} vs bigram {criteria['mean_bigram_top1']:.4f}；"
        f"赢的折 {criteria['folds_winning']}/{len(folds)}；均值差 {mean_delta:+.4f}"
    )
    flags = " ".join(
        f"{k}={'过' if v else '不过'}" for k, v in criteria.items() if isinstance(v, bool)
    )
    print(f"[判据] {flags}")
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
