"""S1：递推电流受控时，循环还有没有价值（判据见 `research/sel/prereg-lm-s1-controlled-rec.md`）。

四臂：`none`、`rec@0.5`（冻结）、`rec@1.0`（冻结）、`rec@0.5-learned`。
除"递推怎么来、电流多大"外，其余与 M0b/S0 完全一致。
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
OUT = REPO / "sweep_results" / "lm_eprop_s1"

UPDATES = 2000
TAU = 20.0
TAU_READ = 8.0
INPUT_STD = 0.12
LR_IN = 1e-3
LR_REC = 1e-3
LR_OUT = 0.5
BIGRAM_K = 0.1

#: 判据（改这里必须改预注册）。
BASELINE_CE, BASELINE_TOP1 = 1.139, 0.717  # S0 的 none 臂
RATIO_TOL = 0.15
WIN_DELTA_TOP1 = 0.02
WIN_DELTA_CE = 0.05
REGRESS_TOL = 0.02

ARMS = (
    ("none", None, False),
    ("rec@0.5", 0.5, False),
    ("rec@1.0", 1.0, False),
    ("rec@0.5-learned", 0.5, True),
)


def build(vocab_size: int, device: torch.device) -> TokenEProp:
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
        recurrence="random",
        generator=torch.Generator().manual_seed(SEED),
        device=device,
    )


def train(
    model: TokenEProp, sequences: list[list[int]], pad: int, *, learn_recurrent: bool, updates: int
) -> dict:
    names = ["w_in", "w_out"] + (["w_rec"] if learn_recurrent else [])
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
            learn_recurrent=learn_recurrent,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S1：受控递推电流")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--tag", default="s1")
    args = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = load_splits(args.data)
    vocab = build_vocab(splits["train"])
    train_set = encode_split(splits["train"], vocab)
    test = encode_split(splits["test"], vocab)
    pad = len(vocab) - 1
    bigram = top1_accuracy(train_counts(train_set, len(vocab)), test, BIGRAM_K)
    calibrate_tokens, _ = make_batches(train_set, order=list(range(BATCH)), pad=pad)
    print(
        f"train {len(train_set)} / test {len(test)} / vocab {len(vocab)} / bigram {bigram:.4f}",
        flush=True,
    )

    arms: dict[str, dict] = {}
    for name, target, learn_recurrent in ARMS:
        print(f"[{name}] 训练中…", flush=True)
        model = build(len(vocab), device)
        calibration = {"target": target, "achieved": 0.0, "total_scale": 1.0, "rounds": 0}
        if target is None:
            model.recurrence = "none"  # 关掉递推
        else:
            calibration = model.calibrate_recurrent(calibrate_tokens.to(device), target=target)
        summary = train(
            model, train_set, pad, learn_recurrent=learn_recurrent, updates=args.updates
        )
        tokens, valid = make_batches(test, order=list(range(len(test))), pad=pad)
        metrics = model.evaluate(
            tokens.to(device), model.new_trace(tokens.shape[0]), valid=valid.to(device)
        )
        arms[name] = {
            **{key: round(value, 4) for key, value in metrics.items()},
            **summary,
            "calibration": calibration,
            "silent_rate": round(model.silent_rate(BATCH), 4),
            "ratio_after_training": round(
                model.measure_current_ratio(calibrate_tokens.to(device)), 4
            )
            if target is not None
            else 0.0,
        }
        print(f"[{name}] {arms[name]}", flush=True)

    none = arms["none"]

    def wins(arm: dict) -> bool:
        return (
            arm["top1"] >= none["top1"] + WIN_DELTA_TOP1 or arm["ce"] <= none["ce"] - WIN_DELTA_CE
        )

    criteria = {
        "s1_0_calibrated": all(
            abs(arms[name]["calibration"]["achieved"] - target) <= RATIO_TOL
            for name, target, _ in ARMS
            if target is not None
        ),
        "s1_1_rec05_wins": wins(arms["rec@0.5"]),
        "s1_2_rec10_wins": wins(arms["rec@1.0"]),
        "s1_3_learned_wins_over_frozen": (
            arms["rec@0.5-learned"]["top1"] >= arms["rec@0.5"]["top1"] + WIN_DELTA_TOP1
            or arms["rec@0.5-learned"]["ce"] <= arms["rec@0.5"]["ce"] - WIN_DELTA_CE
        ),
        "s1_4_no_arm_regresses": all(
            arms[name]["top1"] >= none["top1"] - REGRESS_TOL
            for name, target, _ in ARMS
            if target is not None
        ),
        "none_reproduces": abs(none["top1"] - BASELINE_TOP1) <= 0.01,
    }

    out = {
        "kind": "S1: controlled recurrent current ratio",
        "seed": SEED,
        "config": {
            "n_rec": N_REC,
            "fan_in": FAN_IN,
            "max_len": MAX_LEN,
            "batch": BATCH,
            "updates": args.updates,
            "lr_in": LR_IN,
            "lr_rec": LR_REC,
            "lr_out": LR_OUT,
        },
        "bigram_top1": round(bigram, 4),
        "baseline_s0_none": {"ce": BASELINE_CE, "top1": BASELINE_TOP1},
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
            f"calib {arm['calibration']} silent {arm['silent_rate']}"
        )
    flags = " ".join(
        f"{k}={'过' if v else '不过'}" for k, v in criteria.items() if isinstance(v, bool)
    )
    print(f"[判据] {flags}")
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
