"""A4：换掉读出之后，规模还买得到精度吗（判据见 `prereg-a4-readout-scale.md`）。

一臂：`n_rec = 32,768`、`taps = 28`，其余与 A3 的 `k=28` 一致。
对照不重跑——8,192 / 28 抽头 = 0.917 已在 A3 实测。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from research.brainregions.run_a1 import BATCH, EPOCHS
from research.brainregions.run_a3 import SEED, build, evaluate
from research.eprop import load_smnist

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "sweep_results" / "brainregions_a4"

N_REC = 32768
TAPS = 28
REFERENCE_8K = 0.917
GAIN = 0.02
FLOOR = 0.907
RATE_LOW, RATE_HIGH = 0.06, 0.24
MEMORY_LIMIT = 8000.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A4：32k + 28 抽头")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="a4")
    args = parser.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(SEED)
    train = load_smnist("data", split="train", seed=SEED)
    test = load_smnist("data", split="test", seed=SEED)
    inputs, targets = train.batch(torch.arange(len(train)))
    inputs, targets = inputs.to(device), targets.to(device)
    n_out = int(max(int(train.labels.max()), int(test.labels.max()))) + 1
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = build(int(inputs.shape[2]), n_out, TAPS, n_rec=N_REC).to(device)
    generator = torch.Generator().manual_seed(SEED + 991)
    started = time.perf_counter()
    for _ in range(EPOCHS):
        permutation = torch.randperm(inputs.shape[1], generator=generator).to(device)
        for start in range(0, inputs.shape[1], BATCH):
            index = permutation[start : start + BATCH]
            model.update(inputs[:, index], targets[index])
    elapsed = time.perf_counter() - started
    result = evaluate(model, test, device)
    peak = round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else 0.0

    gain = result["accuracy"] - REFERENCE_8K
    criteria = {
        "c0_firing_rate_in_band": RATE_LOW <= result["firing_rate"] <= RATE_HIGH,
        "c1_scale_pays": gain >= GAIN,
        "c2_fits_memory": peak < MEMORY_LIMIT,
        "c3_not_broken": result["accuracy"] >= FLOOR,
        "gain": round(gain, 4),
    }
    out = {
        "kind": "line A v4: 32k neurons with a 28-tap readout",
        "n_rec": N_REC,
        "taps": TAPS,
        "seed": SEED,
        "epochs": EPOCHS,
        "reference_8k": REFERENCE_8K,
        "accuracy": round(result["accuracy"], 4),
        "firing_rate": round(result["firing_rate"], 4),
        "peak_mb": peak,
        "elapsed_s": round(elapsed, 1),
        "criteria": criteria,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + chr(10), encoding="utf-8", newline=chr(10)
    )
    print(f"[32k/28taps] {out}", flush=True)
    print(
        f"[C-0] 发放率 {result['firing_rate']:.4f} → {'过' if criteria['c0_firing_rate_in_band'] else '作废'} | "
        f"[C-1] 32k−8k = {gain:+.4f} ≥ {GAIN} → {'过' if criteria['c1_scale_pays'] else '不过'} | "
        f"[C-2] {peak:.0f} MiB → {'过' if criteria['c2_fits_memory'] else '不过'} | "
        f"[C-3] {result['accuracy']:.4f} → {'过' if criteria['c3_not_broken'] else '不过'}"
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
