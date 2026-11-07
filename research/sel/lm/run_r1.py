"""R1：三档难度阶梯上的训练（判据见 `prereg-lm-r1-synthetic-ladder.md`）。

每档六折（按生成时写死的 `group`），每折独立建词表、独立训练、独立评测；
基线是同折的字符 bigram（与 `measure_ladder.py` 同一口径）。
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
DATA = Path("D:/AxoData/sel/lm/v1/synth")
OUT = REPO / "sweep_results" / "lm_eprop_r1"

LEVELS = ("E", "M", "H")
UPDATES = 2000
MAX_LEN = 192
MAX_VOCAB = 800
TAU = 20.0
TAU_READ = 8.0
INPUT_STD = 0.12
LR_IN = 1e-3
LR_OUT = 0.5
SEED = 0
BIGRAM_K = 0.5

#: 判据（改这里必须改预注册）。
LOSS_DROP = 0.6
WIN_DELTA = 0.0


def load_groups(level: str, name: str = "") -> dict[int, list[str]]:
    groups: dict[int, list[str]] = defaultdict(list)
    path = DATA / f"{name or level}.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        groups[int(row["group"])].append(f"\x01{row['user']}\x02{row['assistant']}\x03")
    return dict(groups)


def build_vocab(texts: list[str]) -> dict[str, int]:
    counts = Counter(ch for text in texts for ch in text)
    common = [ch for ch, _ in counts.most_common(MAX_VOCAB)]
    return {token: index for index, token in enumerate(["<unk>", *common])}


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


def run_fold(
    train_texts: list[str],
    test_texts: list[str],
    device,
    updates: int,
    *,
    readout_width: int | None = None,
    readout_taus: tuple[float, ...] | None = None,
    eval_every: int = 0,
    n_rec: int = N_REC,
) -> dict:
    torch.manual_seed(SEED)
    vocab = build_vocab(train_texts)
    train = [encode(text, vocab, cap=MAX_LEN) for text in train_texts]
    test = [encode(text, vocab, cap=MAX_LEN) for text in test_texts]
    pad = vocab["<unk>"]
    model = TokenEProp(
        len(vocab),
        n_rec,
        fan_in=FAN_IN,
        tau=TAU,
        tau_read=TAU_READ,
        input_std=INPUT_STD,
        lr_in=LR_IN,
        lr_out=LR_OUT,
        recurrence="none",
        readout_width=readout_width,
        readout_taus=readout_taus,
        generator=torch.Generator().manual_seed(SEED),
        device=device,
    )
    generator = torch.Generator().manual_seed(SEED + 17)
    losses: list[float] = []
    curve: list[dict] = []
    probe_tokens, probe_valid = make_batches(test, order=list(range(len(test))), pad=pad)
    for step in range(updates):
        order = torch.randperm(len(train), generator=generator)[:BATCH].tolist()
        tokens, valid = make_batches(train, order=order, pad=pad)
        result = model.run(
            tokens.to(device), model.new_trace(tokens.shape[0]), valid=valid.to(device)
        )
        losses.append(result["ce"])
        if eval_every and (step + 1) % eval_every == 0:
            probe = model.evaluate(
                probe_tokens.to(device),
                model.new_trace(probe_tokens.shape[0]),
                valid=probe_valid.to(device),
            )
            curve.append(
                {"updates": step + 1, "top1": round(probe["top1"], 4), "ce": round(probe["ce"], 4)}
            )
            print(f"    step {step + 1}: top1 {probe['top1']:.4f} ce {probe['ce']:.3f}", flush=True)
    tokens, valid = probe_tokens, probe_valid
    metrics = model.evaluate(
        tokens.to(device), model.new_trace(tokens.shape[0]), valid=valid.to(device)
    )
    base_top1, base_ce = bigram_accuracy(train, test, len(vocab))
    head = statistics.median(losses[:200])
    tail = statistics.median(losses[-200:])
    return {
        "vocab": len(vocab),
        "top1": round(metrics["top1"], 4),
        "ce": round(metrics["ce"], 4),
        "bigram_top1": round(base_top1, 4),
        "bigram_ce": round(base_ce, 4),
        "delta_top1": round(metrics["top1"] - base_top1, 4),
        "ce_ratio": round(tail / head, 4) if head else None,
        "tokens": metrics["tokens"],
        "readout_dims": int(model.w_out.shape[0]),
        "curve": curve,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R1：三档阶梯")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--levels", default=",".join(LEVELS))
    parser.add_argument("--folds", default="", help="逗号分隔的留出 group；空=全部")
    parser.add_argument("--name", default="", help="语料文件名（不含 .jsonl）；空=用 level")
    parser.add_argument("--readout-width", type=int, default=0, help="0=读全部神经元")
    parser.add_argument(
        "--readout-taus", default="", help="逗号分隔；空=单 EMA（tau_read）。给了就走 R4"
    )
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--n-rec", type=int, default=N_REC)
    parser.add_argument("--tag", default="r1")
    args = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    levels = [item.strip() for item in args.levels.split(",") if item.strip()]
    taus = tuple(float(item) for item in args.readout_taus.split(",") if item.strip()) or None

    report: dict[str, dict] = {}
    fold_ids = [int(item) for item in args.folds.split(",") if item.strip()] if args.folds else None
    for level in levels:
        groups = load_groups(level, args.name)
        held_ids = fold_ids if fold_ids is not None else sorted(groups)
        folds: dict[str, dict] = {}
        for held in held_ids:
            train_texts = [text for g in groups if g != held for text in groups[g]]
            folds[str(held)] = run_fold(
                train_texts,
                groups[held],
                device,
                args.updates,
                readout_width=args.readout_width or None,
                readout_taus=taus,
                eval_every=args.eval_every,
                n_rec=args.n_rec,
            )
        deltas = [fold["delta_top1"] for fold in folds.values()]
        report[level] = {
            "folds": folds,
            "mean_top1": round(statistics.mean(f["top1"] for f in folds.values()), 4),
            "mean_bigram_top1": round(statistics.mean(f["bigram_top1"] for f in folds.values()), 4),
            "mean_delta": round(statistics.mean(deltas), 4),
            "folds_winning": sum(1 for d in deltas if d > 0),
            "loss_falls_all": all(
                f["ce_ratio"] is not None and f["ce_ratio"] < LOSS_DROP for f in folds.values()
            ),
        }
        print(
            f"[{level}] {report[level]['mean_top1']} vs bigram "
            f"{report[level]['mean_bigram_top1']} → Δ {report[level]['mean_delta']:+.4f}，"
            f"赢 {report[level]['folds_winning']}/{len(held_ids)}",
            flush=True,
        )

    deltas_by_level = [report[level]["mean_delta"] for level in levels]
    criteria = {
        "r1_2_loss_falls": all(report[level]["loss_falls_all"] for level in levels),
        "r1_3_monotone": deltas_by_level == sorted(deltas_by_level, reverse=True),
        "r1_4_middle_wins": report.get("M", {}).get("mean_delta", -1.0) > WIN_DELTA
        if "M" in report
        else None,
    }
    out = {
        "kind": "R1: difficulty ladder for the feedforward local-rule LM",
        "seed": SEED,
        "config": {
            # 必须记 args.n_rec：记模块常量会让 --n-rec 的臂在 JSON 里全部冒充 1024
            "n_rec": args.n_rec,
            "fan_in": FAN_IN,
            "max_len": MAX_LEN,
            "max_vocab": MAX_VOCAB,
            "batch": BATCH,
            "updates": args.updates,
            "lr_in": LR_IN,
            "lr_out": LR_OUT,
            "readout_width": args.readout_width or None,
            "readout_taus": list(taus) if taus else None,
        },
        "levels": report,
        "criteria": criteria,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"三档 Δ：{ {level: report[level]['mean_delta'] for level in levels} }")
    flags = " ".join(
        f"{k}={'过' if v else '不过'}" for k, v in criteria.items() if isinstance(v, bool)
    )
    print(f"[判据] {flags}")
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
