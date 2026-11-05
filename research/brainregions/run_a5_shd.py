"""A5：多抽头读出在 SHD 上复现（判据见 `prereg-a5-shd-multitap.md`）。

一臂：8,192 神经元、fan_in 32、taps 28、SHD、10 epoch、batch 8、seed 0。
对照不重跑——S8 的同配置"只读最后一步"是 0.338。
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
OUT = REPO / "sweep_results" / "brainregions_a5"

N_REC = 8192
TAPS = 28
EPOCHS = 10
REFERENCE_LAST_STEP = 0.338
GAIN = 0.03
FLOOR = 0.20
MEMORY_LIMIT = 8000.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A5：SHD + 28 抽头")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="a5")
    parser.add_argument("--reference", type=float, default=REFERENCE_LAST_STEP)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-rec", type=int, default=N_REC)
    args = parser.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train = load_shd("data", split="train", seed=seed)
    test = load_shd("data", split="test", seed=seed)
    inputs, targets = train.batch(torch.arange(len(train)))
    inputs, targets = inputs.to(device), targets.to(device)
    n_out = int(max(int(train.labels.max()), int(test.labels.max()))) + 1
    model = build(int(inputs.shape[2]), n_out, TAPS, n_rec=args.n_rec, seed=seed).to(device)
    start = time.perf_counter()
    # 与 A3 同一口径：生成器只在循环外建一次，每个 epoch 抽不同的排列。
    generator = torch.Generator().manual_seed(seed + 991)
    for _ in range(EPOCHS):
        permutation = torch.randperm(inputs.shape[1], generator=generator)
        for offset in range(0, inputs.shape[1], BATCH):
            index = permutation[offset : offset + BATCH].to(device)
            model.update(inputs[:, index], targets[index])
        print(f"epoch done, elapsed {time.perf_counter() - start:.0f}s", flush=True)
    elapsed = time.perf_counter() - start
    result = evaluate(model, test, device)
    peak = round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else 0.0

    gain = result["accuracy"] - args.reference
    criteria = {
        "f0_not_broken": result["accuracy"] > FLOOR,
        "f1_reproduces": gain >= GAIN,
        "f2_not_worse": result["accuracy"] >= REFERENCE_LAST_STEP,
        "f3_fits_memory": peak < MEMORY_LIMIT,
        "gain": round(gain, 4),
    }
    out = {
        "kind": "line A v5: multi-tap readout on SHD",
        "n_rec": args.n_rec,
        "fan_in": FAN_IN,
        "taps": TAPS,
        "epochs": EPOCHS,
        "seed": seed,
        "test_samples": TEST_SAMPLES,
        "reference_last_step": args.reference,
        "accuracy": round(result["accuracy"], 4),
        "firing_rate": round(result["firing_rate"], 4),
        "peak_mb": peak,
        "elapsed_s": round(elapsed, 1),
        "criteria": criteria,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"[SHD 8k/28taps] {out}")
    print(
        f"[F-0] {result['accuracy']:.4f} > {FLOOR} -> {'过' if criteria['f0_not_broken'] else '不过'} | "
        f"[F-1] 相对 0.338 {gain:+.4f} >= {GAIN} -> {'过' if criteria['f1_reproduces'] else '不过'} | "
        f"[F-2] {'过' if criteria['f2_not_worse'] else '不过'} | "
        f"[F-3] {peak:.0f} MiB -> {'过' if criteria['f3_fits_memory'] else '不过'}"
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
