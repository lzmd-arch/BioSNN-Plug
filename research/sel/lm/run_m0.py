"""M0：局部规则能不能学 next-token（判据见 `research/sel/prereg-lm-eprop.md`）。

两臂同数据、同种子、同预算：**学习 w_rec** 与 **冻结 w_rec**；外加 bigram 分母。
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
from research.sel.student import build_vocab, encode, load_splits

REPO = Path(__file__).resolve().parents[3]
DATA = Path("D:/AxoData/sel/pairs/v1_amend3/pairs.jsonl")
OUT = REPO / "sweep_results" / "lm_eprop_m0"

N_REC = 1024
FAN_IN = 32
MAX_LEN = 64
BATCH = 16
UPDATES = 2000
SEED = 0
TAU = 20.0
TAU_READ = 8.0
INPUT_STD = 0.12
LR_IN = 2e-4
LR_REC = 1e-4
LR_OUT = 0.5
BIGRAM_K = 0.1

#: 判据（改这里必须改预注册）。
BIGRAM_EXPECT = 0.6457
BIGRAM_TOL = 0.02
CE_INIT_LOW, CE_INIT_HIGH = 5.0, 6.2
RATE_LOW, RATE_HIGH = 0.01, 0.30
TOP1_MIN = 0.70
REC_GAIN = 0.95
TRAIN_CE_DROP = 0.5


def encode_split(texts: list[str], vocab: dict[str, int]) -> list[list[int]]:
    return [encode(text, vocab, max_len=MAX_LEN) for text in texts]


def make_batches(
    sequences: list[list[int]], *, order: list[int], pad: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    width = max(len(sequences[i]) for i in order)
    tokens = torch.full((len(order), width), pad, dtype=torch.long)
    valid = torch.zeros((len(order), width), dtype=torch.bool)
    for row, index in enumerate(order):
        sequence = sequences[index]
        tokens[row, : len(sequence)] = torch.tensor(sequence)
        valid[row, : len(sequence)] = True
    return tokens, valid


def run_arm(
    sequences: list[list[int]],
    vocab_size: int,
    *,
    learn_recurrent: bool,
    device: torch.device,
    updates: int = UPDATES,
) -> TokenEProp:
    torch.manual_seed(SEED)
    model = TokenEProp(
        vocab_size,
        N_REC,
        fan_in=FAN_IN,
        tau=TAU,
        tau_read=TAU_READ,
        input_std=INPUT_STD,
        lr_in=LR_IN,
        lr_rec=LR_REC,
        lr_out=LR_OUT,
        generator=torch.Generator().manual_seed(SEED),
        device=device,
    )
    pad = vocab_size - 1
    generator = torch.Generator().manual_seed(SEED + 17)
    losses: list[float] = []
    rates: list[float] = []
    step = 0
    started = time.perf_counter()
    while step < updates:
        order = torch.randperm(len(sequences), generator=generator)[:BATCH].tolist()
        tokens, valid = make_batches(sequences, order=order, pad=pad)
        result = model.run(
            tokens.to(device),
            model.new_trace(tokens.shape[0]),
            valid=valid.to(device),
            learn_recurrent=learn_recurrent,
        )
        losses.append(result["ce"])
        rates.append(result["firing_rate"])
        step += 1
        if step % 500 == 0:
            print(
                f"  step {step} ce {result['ce']:.3f} rate {result['firing_rate']:.3f} "
                f"({time.perf_counter() - started:.0f}s)",
                flush=True,
            )
    head = statistics.median(losses[:200])
    tail = statistics.median(losses[-200:])
    model.training_summary = {  # type: ignore[attr-defined]
        "ce_head": round(head, 4),
        "ce_tail": round(tail, 4),
        "ce_ratio": round(tail / head, 4) if head else None,
        "train_firing_rate": round(statistics.median(rates), 4),
        "seconds": round(time.perf_counter() - started, 1),
    }
    return model


def evaluate_model(
    model: TokenEProp, sequences: list[list[int]], vocab_size: int, device: torch.device
) -> dict:
    pad = vocab_size - 1
    order = list(range(len(sequences)))
    tokens, valid = make_batches(sequences, order=order, pad=pad)
    return model.evaluate(
        tokens.to(device), model.new_trace(tokens.shape[0]), valid=valid.to(device)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M0：局部规则 next-token")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--tag", default="m0")
    args = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = load_splits(args.data)
    vocab = build_vocab(splits["train"])
    train = encode_split(splits["train"], vocab)
    test = encode_split(splits["test"], vocab)
    print(f"train {len(train)} / test {len(test)} / vocab {len(vocab)}", flush=True)

    counts = train_counts(train, len(vocab))
    bigram = top1_accuracy(counts, test, BIGRAM_K)

    arms = {}
    for name, learn_recurrent in (("learned", True), ("frozen", False)):
        print(f"[{name}] 训练中…", flush=True)
        model = run_arm(
            train, len(vocab), learn_recurrent=learn_recurrent, device=device, updates=args.updates
        )
        metrics = evaluate_model(model, test, len(vocab), device)
        arms[name] = {**metrics, **model.training_summary}  # type: ignore[attr-defined]
        print(f"[{name}] {arms[name]}", flush=True)

    initial = TokenEProp(
        len(vocab),
        N_REC,
        fan_in=FAN_IN,
        generator=torch.Generator().manual_seed(SEED),
        device=device,
    )
    initial_metrics = evaluate_model(initial, test, len(vocab), device)

    learned, frozen = arms["learned"], arms["frozen"]
    criteria = {
        "m0_0_data_matches": abs(bigram - BIGRAM_EXPECT) <= BIGRAM_TOL,
        "m0_0b_init_ce_in_band": CE_INIT_LOW <= initial_metrics["ce"] <= CE_INIT_HIGH,
        "m0_0c_firing_in_band": RATE_LOW <= arms["learned"]["firing_rate"] <= RATE_HIGH,
        "m0_1_local_rule_learns": learned["top1"] >= TOP1_MIN,
        "m0_2_recurrent_learns": learned["ce"] <= REC_GAIN * frozen["ce"],
        "m0_3_loss_falls": learned["ce_ratio"] is not None and learned["ce_ratio"] < TRAIN_CE_DROP,
    }
    out = {
        "kind": "local-rule next-token M0 (TokenEProp)",
        "seed": SEED,
        "config": {
            "n_rec": N_REC,
            "fan_in": FAN_IN,
            "max_len": MAX_LEN,
            "batch": BATCH,
            "updates": args.updates,
            "tau": TAU,
            "tau_read": TAU_READ,
            "input_std": INPUT_STD,
            "lr_in": LR_IN,
            "lr_rec": LR_REC,
            "lr_out": LR_OUT,
            "device": str(device),
        },
        "vocab": len(vocab),
        "bigram_top1": round(bigram, 4),
        "bigram_k": BIGRAM_K,
        "initial": {key: round(value, 4) for key, value in initial_metrics.items()},
        "arms": arms,
        "criteria": criteria,
        "m0_pass": bool(criteria["m0_1_local_rule_learns"] and criteria["m0_2_recurrent_learns"]),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"[bigram] {bigram:.4f}（期望 {BIGRAM_EXPECT}±{BIGRAM_TOL}）")
    print(
        f"[init] ce {initial_metrics['ce']:.3f}（ln V = {torch.log(torch.tensor(float(len(vocab)))).item():.3f}）"
    )
    flags = " ".join(f"{name}={'过' if value else '不过'}" for name, value in criteria.items())
    print(f"[判据] {flags}")
    print(f"[M0] {'通过' if out['m0_pass'] else '不通过'}")
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
