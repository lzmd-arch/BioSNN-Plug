"""A9：A 线去掉递推之后，绝对数字是不是被低估了（判据见 `prereg-a9-no-recurrence.md`）。

与 A3 的 `k=28` 臂逐字相同，唯一改动是 `w_rec` 置零且冻结。
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from research.brainregions.run_a1 import BATCH, EPOCHS
from research.brainregions.run_a3 import build, evaluate
from research.eprop import load_smnist

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "sweep_results" / "brainregions_a9"

TAPS = 28
SEEDS = (0, 1, 2)
#: A3 的 k=28 三种子（有递推），用来做逐种子对照。
A3_BY_SEED = {0: 0.917, 1: 0.873, 2: 0.906}
A3_MEDIAN = 0.899
RATE_LOW, RATE_HIGH = 0.01, 0.30
GREEN, GREY, RED = 0.92, 0.90, 0.90


def run_seed(seed: int, device: str, epochs: int) -> dict:
    torch.manual_seed(seed)
    train = load_smnist("data", split="train", seed=seed)
    test = load_smnist("data", split="test", seed=seed)
    inputs, targets = train.batch(torch.arange(len(train)))
    inputs, targets = inputs.to(device), targets.to(device)
    n_out = int(max(int(train.labels.max()), int(test.labels.max()))) + 1
    model = build(int(inputs.shape[2]), n_out, TAPS, seed=seed).to(device)
    with torch.no_grad():
        model.w_rec.zero_()  # 去掉递推
    generator = torch.Generator().manual_seed(seed + 991)
    for _ in range(epochs):
        permutation = torch.randperm(inputs.shape[1], generator=generator)
        for offset in range(0, inputs.shape[1], BATCH):
            index = permutation[offset : offset + BATCH].to(device)
            model.update(inputs[:, index], targets[index], learn_recurrent=False)
    result = evaluate(model, test, device)
    return {
        "accuracy": round(result["accuracy"], 4),
        "firing_rate": round(result["firing_rate"], 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A9：无递推的 A 线")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--tag", default="a9")
    args = parser.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())

    rows: dict[str, dict] = {}
    for seed in seeds:
        rows[str(seed)] = run_seed(seed, device, args.epochs)
        print(f"[seed {seed}] {rows[str(seed)]}", flush=True)

    accuracies = [rows[str(seed)]["accuracy"] for seed in seeds]
    median = statistics.median(accuracies)
    deltas = {
        str(seed): round(rows[str(seed)]["accuracy"] - A3_BY_SEED[seed], 4)
        for seed in seeds
        if seed in A3_BY_SEED
    }
    rates_ok = all(RATE_LOW <= rows[str(seed)]["firing_rate"] <= RATE_HIGH for seed in seeds)
    criteria = {
        "a9_0_rate_in_band": rates_ok,
        "a9_1_meaningful_gain": median >= GREEN,
        "a9_2_limited_gain": GREY <= median < GREEN,
        "a9_3_hypothesis_refuted": median < RED,
        "median": round(median, 4),
        "per_seed_delta_vs_a3": deltas,
    }
    out = {
        "kind": "A9: line A with the recurrence removed",
        "seeds": list(seeds),
        "taps": TAPS,
        "epochs": args.epochs,
        "n_rec": 8192,
        "fan_in": 32,
        "a3_reference": {"by_seed": A3_BY_SEED, "median": A3_MEDIAN},
        "rows": rows,
        "criteria": criteria,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"acc {accuracies} 中位 {median:.4f}（A3 中位 {A3_MEDIAN}）逐种子 {deltas}")
    flags = " ".join(
        f"{k}={'过' if v else '不过'}" for k, v in criteria.items() if isinstance(v, bool)
    )
    print(f"[判据] {flags}")
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
