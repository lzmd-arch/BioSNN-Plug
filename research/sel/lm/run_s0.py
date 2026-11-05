"""S0：SSM（结构化递推）能不能赢过随机循环（判据见 `research/sel/prereg-lm-s0-ssm.md`）。

三臂：`none`（无递推）、`random`（冻结随机 w_rec，M0b 同款）、`ssm`（冻结对角 SSM）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from research.sel.baseline_bigram import top1_accuracy, train_counts
from research.sel.lm.model import TokenEProp
from research.sel.lm.run_m0 import BATCH, FAN_IN, MAX_LEN, N_REC, SEED, encode_split, make_batches
from research.sel.student import build_vocab, load_splits

REPO = Path(__file__).resolve().parents[3]
DATA = Path("D:/AxoData/sel/pairs/v1_amend3/pairs.jsonl")
OUT = REPO / "sweep_results" / "lm_eprop_s0"

UPDATES = 2000
TAU = 20.0
TAU_READ = 8.0
INPUT_STD = 0.12
LR_IN = 1e-3
LR_REC = 1e-3
LR_OUT = 0.5
N_STATE = 32
BIGRAM_K = 0.1

#: 判据（改这里必须改预注册）。
BIGRAM_CE, BIGRAM_TOP1 = 1.4234, 0.6457
RANDOM_CE, RANDOM_TOP1 = 1.741, 0.566
RATE_LOW, RATE_HIGH = 0.01, 0.30
WIN_DELTA_TOP1 = 0.02
WIN_DELTA_CE = 0.05
BEAT_BIGRAM_CE, BEAT_BIGRAM_TOP1 = 1.45, 0.65


def build(vocab_size: int, recurrence: str) -> TokenEProp:
    return TokenEProp(
        vocab_size,
        N_REC,
        fan_in=FAN_IN,
        tau=TAU,
        tau_read=TAU_READ,
        input_std=INPUT_STD,
        lr_in=LR_IN,
        lr_rec=LR_REC,
        lr_out=LR_OUT,
        recurrence=recurrence,
        n_state=N_STATE,
        generator=torch.Generator().manual_seed(SEED),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


def train_arm(
    model: TokenEProp, sequences: list[list[int]], pad: int, *, updates: int = UPDATES
) -> dict:
    names = ["w_out"] if model.recurrence == "ssm" else ["w_in", "w_rec", "w_out"]
    if model.recurrence == "ssm":
        names = ["E", "C", "D", "w_out"]
    snapshot = {name: getattr(model, name).clone() for name in names}
    generator = torch.Generator().manual_seed(SEED + 17)
    losses: list[float] = []
    rates: list[float] = []
    started = time.perf_counter()
    for _ in range(updates):
        order = torch.randperm(len(sequences), generator=generator)[:BATCH].tolist()
        tokens, valid = make_batches(sequences, order=order, pad=pad)
        result = model.run(
            tokens.to(model.device),
            model.new_trace(tokens.shape[0]),
            valid=valid.to(model.device),
        )
        losses.append(result["ce"])
        rates.append(result["firing_rate"])
    moved = {
        name: round(float((getattr(model, name) - base).norm() / base.norm()), 6)
        for name, base in snapshot.items()
    }
    return {
        "ce_head": round(statistics.median(losses[:200]), 4),
        "ce_tail": round(statistics.median(losses[-200:]), 4),
        "ce_ratio": round(statistics.median(losses[-200:]) / statistics.median(losses[:200]), 4),
        "train_firing_rate": round(statistics.median(rates), 4),
        "relative_change": moved,
        "seconds": round(time.perf_counter() - started, 1),
    }


def state_norm(model: TokenEProp, sequences: list[list[int]], pad: int) -> float | None:
    if model.recurrence != "ssm":
        return None
    tokens, _ = make_batches(sequences, order=list(range(min(16, len(sequences)))), pad=pad)
    trace = model.new_trace(tokens.shape[0])
    norms: list[float] = []
    for t in range(tokens.shape[1] - 1):
        model.step(
            tokens[:, t].to(model.device), tokens[:, t + 1].to(model.device), trace, update=False
        )
        norms.append(float(trace.h.norm(dim=1).mean()))
    return round(statistics.median(norms), 4)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S0：SSM vs 随机循环")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--tag", default="s0")
    args = parser.parse_args(argv)

    splits = load_splits(args.data)
    vocab = build_vocab(splits["train"])
    train = encode_split(splits["train"], vocab)
    test = encode_split(splits["test"], vocab)
    pad = len(vocab) - 1
    bigram = top1_accuracy(train_counts(train, len(vocab)), test, BIGRAM_K)
    print(
        f"train {len(train)} / test {len(test)} / vocab {len(vocab)} / bigram {bigram:.4f}",
        flush=True,
    )

    arms: dict[str, dict] = {}
    for name in ("none", "random", "ssm"):
        print(f"[{name}] 训练中…", flush=True)
        model = build(len(vocab), name)
        summary = train_arm(model, train, pad, updates=args.updates)
        tokens, valid = make_batches(test, order=list(range(len(test))), pad=pad)
        metrics = model.evaluate(
            tokens.to(model.device), model.new_trace(tokens.shape[0]), valid=valid.to(model.device)
        )
        arms[name] = {
            **{key: round(value, 4) for key, value in metrics.items()},
            **summary,
            "state_norm": state_norm(model, train, pad),
        }
        print(f"[{name}] {arms[name]}", flush=True)

    none, random, ssm = arms["none"], arms["random"], arms["ssm"]
    criteria = {
        "s0_0_preconditions": RATE_LOW <= random["firing_rate"] <= RATE_HIGH
        and RATE_LOW <= ssm["firing_rate"] <= RATE_HIGH,
        "s0_1_ssm_beats_random": (
            ssm["top1"] >= random["top1"] + WIN_DELTA_TOP1
            or ssm["ce"] <= random["ce"] - WIN_DELTA_CE
        ),
        "s0_2_beats_bigram": ssm["top1"] >= BEAT_BIGRAM_TOP1 or ssm["ce"] <= BEAT_BIGRAM_CE,
        "s0_3_diagnostics": ssm["state_norm"] is not None,
        "delta_top1_vs_random": round(ssm["top1"] - random["top1"], 4),
        "delta_ce_vs_random": round(ssm["ce"] - random["ce"], 4),
        "none_firing_low": none["firing_rate"] < 0.005,
    }
    out = {
        "kind": "S0: structured recurrence (diagonal SSM) vs frozen random recurrence",
        "seed": SEED,
        "config": {
            "n_rec": N_REC,
            "fan_in": FAN_IN,
            "n_state": N_STATE,
            "max_len": MAX_LEN,
            "batch": BATCH,
            "updates": args.updates,
            "lr_in": LR_IN,
            "lr_rec": LR_REC,
            "lr_out": LR_OUT,
        },
        "baselines": {
            "bigram_ce": BIGRAM_CE,
            "bigram_top1": BIGRAM_TOP1,
            "m0b_random_ce": RANDOM_CE,
            "m0b_random_top1": RANDOM_TOP1,
        },
        "bigram_top1_recomputed": round(bigram, 4),
        "arms": arms,
        "criteria": criteria,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    for name, arm in arms.items():
        print(
            f"[{name}] ce {arm['ce']:.3f} top1 {arm['top1']:.3f} rate {arm['firing_rate']:.3f} "
            f"moved {arm['relative_change']}"
        )
    flags = " ".join(
        f"{k}={'过' if v else '不过'}" for k, v in criteria.items() if isinstance(v, bool)
    )
    print(f"[判据] {flags}")
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
