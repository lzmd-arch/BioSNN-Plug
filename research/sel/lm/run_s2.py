"""S2：0.717 是不是运气——纯前馈 LIF + 局部读出，五个种子（判据见 `prereg-lm-s2-seeds.md`）。"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from research.sel.baseline_bigram import top1_accuracy, train_counts
from research.sel.lm.model import TokenEProp
from research.sel.lm.run_m0 import BATCH, FAN_IN, MAX_LEN, N_REC, encode_split, make_batches
from research.sel.student import build_vocab, load_splits

REPO = Path(__file__).resolve().parents[3]
DATA = Path("D:/AxoData/sel/pairs/v1_amend3/pairs.jsonl")
OUT = REPO / "sweep_results" / "lm_eprop_s2"

SEEDS = (0, 1, 2, 3, 4)
UPDATES = 2000
TAU = 20.0
TAU_READ = 8.0
INPUT_STD = 0.12
LR_IN = 1e-3
LR_OUT = 0.5
BIGRAM_K = 0.1

#: 判据（改这里必须改预注册）。
SEED0_TOP1 = 0.7166
SEED0_TOL = 0.01
MEAN_TOP1_MIN = 0.70
STD_TOP1_MAX = 0.02
BIGRAM_CE, BIGRAM_TOP1 = 1.423, 0.646


def run_seed(seed: int, train_set, test, vocab_size: int, pad: int, device, updates: int) -> dict:
    torch.manual_seed(seed)
    model = TokenEProp(
        vocab_size,
        N_REC,
        fan_in=FAN_IN,
        tau=TAU,
        tau_read=TAU_READ,
        input_std=INPUT_STD,
        lr_in=LR_IN,
        lr_out=LR_OUT,
        recurrence="none",
        generator=torch.Generator().manual_seed(seed),
        device=device,
    )
    generator = torch.Generator().manual_seed(seed + 17)
    for _ in range(updates):
        order = torch.randperm(len(train_set), generator=generator)[:BATCH].tolist()
        tokens, valid = make_batches(train_set, order=order, pad=pad)
        model.run(tokens.to(device), model.new_trace(tokens.shape[0]), valid=valid.to(device))
    tokens, valid = make_batches(test, order=list(range(len(test))), pad=pad)
    metrics = model.evaluate(
        tokens.to(device), model.new_trace(tokens.shape[0]), valid=valid.to(device)
    )
    return {key: round(value, 4) for key, value in metrics.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S2：五种子")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--tag", default="s2")
    args = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = load_splits(args.data)
    vocab = build_vocab(splits["train"])
    train_set = encode_split(splits["train"], vocab)
    test = encode_split(splits["test"], vocab)
    pad = len(vocab) - 1
    bigram = top1_accuracy(train_counts(train_set, len(vocab)), test, BIGRAM_K)

    rows: dict[str, dict] = {}
    for seed in SEEDS:
        rows[str(seed)] = run_seed(seed, train_set, test, len(vocab), pad, device, args.updates)
        print(f"[seed {seed}] {rows[str(seed)]}", flush=True)

    top1 = [rows[str(s)]["top1"] for s in SEEDS]
    ces = [rows[str(s)]["ce"] for s in SEEDS]
    mean_top1 = statistics.mean(top1)
    std_top1 = statistics.stdev(top1)
    criteria = {
        "s2_0_seed0_reproduces": abs(top1[0] - SEED0_TOP1) <= SEED0_TOL,
        "s2_1_mean_holds": mean_top1 >= MEAN_TOP1_MIN,
        "s2_2_stable": std_top1 <= STD_TOP1_MAX,
        "s2_3_every_seed_beats_bigram": all(value > BIGRAM_TOP1 for value in top1),
        "s2_4_ce_beats_bigram": statistics.mean(ces) < BIGRAM_CE,
        "mean_top1": round(mean_top1, 4),
        "std_top1": round(std_top1, 4),
        "min_top1": round(min(top1), 4),
        "max_top1": round(max(top1), 4),
    }
    out = {
        "kind": "S2: five seeds for the feedforward local-rule LM",
        "seeds": list(SEEDS),
        "config": {
            "n_rec": N_REC,
            "fan_in": FAN_IN,
            "max_len": MAX_LEN,
            "batch": BATCH,
            "updates": args.updates,
            "lr_in": LR_IN,
            "lr_out": LR_OUT,
            "recurrence": "none",
        },
        "bigram": {"top1": BIGRAM_TOP1, "ce": BIGRAM_CE, "recomputed_top1": round(bigram, 4)},
        "rows": rows,
        "criteria": criteria,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"top1 五种：{top1}  均值 {mean_top1:.4f}  std {std_top1:.4f}")
    flags = " ".join(
        f"{k}={'过' if v else '不过'}" for k, v in criteria.items() if isinstance(v, bool)
    )
    print(f"[判据] {flags}")
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
