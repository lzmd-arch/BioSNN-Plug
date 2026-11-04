"""A6：检验"多抽头增益 ∝ 看不见的比例"（判据见 `prereg-a6-decay-control.md`）。

SHD、8,192、10 epoch，只把读出滤波的衰减从 0.95 调到 0.99，跑 taps=1 与 taps=28 两臂。
参照（decay 0.95）：taps=1 是 0.338（S8）、taps=28 是 0.759（A5）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from research.brainregions.run_a1 import BATCH, FAN_IN, TEST_SAMPLES
from research.brainregions.run_a3 import SEED, build, evaluate
from research.eprop import load_shd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "sweep_results" / "brainregions_a6"

N_REC = 8192
EPOCHS = 10
DECAY = 0.99
ARMS = (1, 28)
FLOOR = 0.20
HALF_GAIN = 0.421 / 2


def run_arm(taps: int, device: str) -> dict:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train = load_shd("data", split="train", seed=SEED)
    test = load_shd("data", split="test", seed=SEED)
    inputs, targets = train.batch(torch.arange(len(train)))
    inputs, targets = inputs.to(device), targets.to(device)
    n_out = int(max(int(train.labels.max()), int(test.labels.max()))) + 1
    model = build(int(inputs.shape[2]), n_out, taps, decay_out=DECAY).to(device)
    generator = torch.Generator().manual_seed(SEED + 991)
    start = time.perf_counter()
    for _ in range(EPOCHS):
        permutation = torch.randperm(inputs.shape[1], generator=generator)
        for offset in range(0, inputs.shape[1], BATCH):
            index = permutation[offset : offset + BATCH].to(device)
            model.update(inputs[:, index], targets[index])
        print(f"  taps={taps} epoch {time.perf_counter() - start:.0f}s", flush=True)
    result = evaluate(model, test, device)
    return {
        "taps": taps,
        "accuracy": round(result["accuracy"], 4),
        "firing_rate": round(result["firing_rate"], 4),
        "elapsed_s": round(time.perf_counter() - start, 1),
        "peak_mb": (
            round(torch.cuda.max_memory_allocated() / 2**20, 1)
            if torch.cuda.is_available()
            else 0.0
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A6：decay_out 对照")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="a6")
    args = parser.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = {f"taps_{taps}": run_arm(taps, device) for taps in ARMS}
    print(f"[A6] {rows}", flush=True)
    single, multi = rows["taps_1"]["accuracy"], rows["taps_28"]["accuracy"]
    gain = multi - single
    criteria = {
        "k0_both_learn": min(single, multi) > FLOOR,
        "k1_gain_shrinks": gain < HALF_GAIN,
        "k2_still_positive": gain > 0.0,
        "gain": round(gain, 4),
        "reference_gain_at_decay_095": 0.421,
    }
    out = {
        "kind": "line A v6: readout decay control on SHD",
        "n_rec": N_REC,
        "fan_in": FAN_IN,
        "epochs": EPOCHS,
        "seed": SEED,
        "decay_out": DECAY,
        "test_samples": TEST_SAMPLES,
        "arms": rows,
        "criteria": criteria,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"[K-0] {'过' if criteria['k0_both_learn'] else '不过'} | "
        f"[K-1] 增益 {gain:+.4f} < {HALF_GAIN:.4f} -> {'过' if criteria['k1_gain_shrinks'] else '不过'} | "
        f"[K-2] {'过' if criteria['k2_still_positive'] else '不过'}"
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
