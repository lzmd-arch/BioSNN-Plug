"""线 A v3：多抽头读出吃不吃得回 A2 的 9.7pp（判据见 `prereg-a3-multitap.md`）。

三档抽头 ``k ∈ {1, 7, 28}``，同一条管线、同一组种子；只有读出宽度变。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from research.brainregions.model import RegionSpec
from research.brainregions.multitap import MultiTapEProp
from research.brainregions.run_a1 import BATCH, EPOCHS, FAN_IN, N_REC, PREDICT_CHUNK, TEST_SAMPLES
from research.eprop import load_smnist

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "sweep_results" / "brainregions_a3"

SEED = 0
TAPS = (1, 7, 28)

#: 判据（改这里必须改预注册）。
A3_BASELINE = 0.759
A3_BASELINE_TOLERANCE = 0.01
A3_GAIN = 0.05
A3_TIME_RATIO = 1.5


def build(n_in: int, n_out: int, taps: int) -> MultiTapEProp:
    return MultiTapEProp(
        n_in,
        N_REC,
        n_out,
        taps=taps,
        fan_in=FAN_IN,
        regions=[RegionSpec("uniform", N_REC, tau=20.0, threshold=0.62)],
        intra_fraction=0.0,
        generator=torch.Generator().manual_seed(SEED),
    )


def evaluate(model: MultiTapEProp, test, device: str) -> dict:
    correct = total = fired = seen = 0
    for start in range(0, TEST_SAMPLES, PREDICT_CHUNK):
        stop = min(start + PREDICT_CHUNK, TEST_SAMPLES)
        inputs, labels = test.batch(torch.arange(start, stop))
        predicted = model.predict(inputs.to(device)).cpu()
        correct += int((predicted == labels).sum())
        total += len(labels)
        with torch.no_grad():
            spikes, _ = model.simulate(inputs.to(device))
        fired += int(spikes.sum())
        seen += spikes.numel()
    return {"accuracy": correct / total, "firing_rate": fired / seen}


def run_once(taps: int, device: str) -> dict:
    torch.manual_seed(SEED)
    train = load_smnist("data", split="train", seed=SEED)
    test = load_smnist("data", split="test", seed=SEED)
    inputs, targets = train.batch(torch.arange(len(train)))
    inputs, targets = inputs.to(device), targets.to(device)
    n_out = int(max(int(train.labels.max()), int(test.labels.max()))) + 1
    model = build(int(inputs.shape[2]), n_out, taps).to(device)
    generator = torch.Generator().manual_seed(SEED + 991)
    started = time.perf_counter()
    for _ in range(EPOCHS):
        permutation = torch.randperm(inputs.shape[1], generator=generator).to(device)
        for start in range(0, inputs.shape[1], BATCH):
            index = permutation[start : start + BATCH]
            model.update(inputs[:, index], targets[index])
    elapsed = time.perf_counter() - started
    return {"elapsed_s": round(elapsed, 1), **evaluate(model, test, device)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="线 A v3：多抽头读出")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="a3")
    args = parser.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows: dict[str, dict] = {}
    for taps in TAPS:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        result = run_once(taps, device)
        if torch.cuda.is_available():
            result["peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        rows[f"taps_{taps}"] = result
        print(f"[k={taps}] {result}", flush=True)

    base, full = rows["taps_1"], rows["taps_28"]
    gain = full["accuracy"] - base["accuracy"]
    time_ratio = full["elapsed_s"] / base["elapsed_s"]
    criteria = {
        "a3_0_baseline_reproduces": abs(base["accuracy"] - A3_BASELINE) <= A3_BASELINE_TOLERANCE,
        "a3_1_taps_pay": gain >= A3_GAIN,
        "a3_2_cost_acceptable": time_ratio <= A3_TIME_RATIO,
        "gain": round(gain, 4),
        "time_ratio": round(time_ratio, 3),
    }
    out = {
        "kind": "line A v3: multi-tap readout",
        "seed": SEED,
        "n_rec": N_REC,
        "fan_in": FAN_IN,
        "epochs": EPOCHS,
        "taps": list(TAPS),
        "arms": rows,
        "criteria": criteria,
        "a3_ready": bool(criteria["a3_0_baseline_reproduces"] and criteria["a3_1_taps_pay"]),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"[A3-0] 基准 {base['accuracy']:.3f} vs {A3_BASELINE} → "
        f"{'过' if criteria['a3_0_baseline_reproduces'] else '不过'} | "
        f"[A3-1] k=28 − k=1 = {gain:+.3f} ≥ {A3_GAIN} → "
        f"{'过' if criteria['a3_1_taps_pay'] else '不过'} | "
        f"[A3-2] 耗时 {time_ratio:.2f}× ≤ {A3_TIME_RATIO}× → "
        f"{'过' if criteria['a3_2_cost_acceptable'] else '不过'}"
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
