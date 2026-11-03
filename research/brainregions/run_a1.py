"""线 A v1：脑区特化 vs 均匀（判据见 `prereg-brain-region-specialization.md`）。

两臂**只有组织不同**：同样的 8,192 神经元、同样的 fan_in=32、同样的数据与预算。
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from research.brainregions.model import RegionEProp, RegionSpec
from research.eprop import exp_convolve, load_smnist

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "sweep_results" / "brainregions_a1"

N_REC = 8192
REGION_SIZE = 1024
EPOCHS = 2
BATCH = 8
FAN_IN = 32
SEEDS = (0, 1, 2)
TEST_SAMPLES = 1000
PREDICT_CHUNK = 100

#: 判据（改这里必须改预注册）。
H_A0_MIN = 0.60
H_A1_GAP = 0.03

#: 四种脑区原型，各两个区（8 × 1024 = 8,192）。
TYPES = (
    ("pfc", 40.0, 0.62),
    ("hippocampus", 10.0, 0.50),
    ("amygdala", 20.0, 0.62),
    ("cerebellum", 8.0, 0.70),
)


def uniform_regions() -> list[RegionSpec]:
    return [RegionSpec("uniform", N_REC, tau=20.0, threshold=0.62)]


def specialized_regions() -> list[RegionSpec]:
    return [
        RegionSpec(f"{name}-{index}", REGION_SIZE, tau=tau, threshold=threshold)
        for name, tau, threshold in TYPES
        for index in (0, 1)
    ]


def evaluate(model: RegionEProp, test, device: str) -> dict:
    correct = total = 0
    fired = seen = 0
    for start in range(0, TEST_SAMPLES, PREDICT_CHUNK):
        stop = min(start + PREDICT_CHUNK, TEST_SAMPLES)
        inputs, labels = test.batch(torch.arange(start, stop))
        inputs = inputs.to(device)
        with torch.no_grad():
            spikes, _ = model.simulate(inputs)
            logits = exp_convolve(spikes, model.decay_out)[-1] @ model.w_out
        predicted = logits.argmax(dim=-1).cpu()
        correct += int((predicted == labels).sum())
        total += len(labels)
        fired += int(spikes.sum())
        seen += spikes.numel()
    return {"accuracy": correct / total, "firing_rate": fired / seen}


def run_once(regions: list[RegionSpec], intra: float, seed: int, device: str) -> dict:
    torch.manual_seed(seed)
    train = load_smnist("data", split="train", seed=seed)
    test = load_smnist("data", split="test", seed=seed)
    inputs, targets = train.batch(torch.arange(len(train)))
    inputs, targets = inputs.to(device), targets.to(device)
    n_out = int(max(int(train.labels.max()), int(test.labels.max()))) + 1
    model = RegionEProp(
        int(inputs.shape[2]),
        N_REC,
        n_out,
        fan_in=FAN_IN,
        regions=regions,
        intra_fraction=intra,
        generator=torch.Generator().manual_seed(seed),
    ).to(device)
    generator = torch.Generator().manual_seed(seed + 991)
    for _ in range(EPOCHS):
        permutation = torch.randperm(inputs.shape[1], generator=generator).to(device)
        for start in range(0, inputs.shape[1], BATCH):
            index = permutation[start : start + BATCH]
            model.update(inputs[:, index], targets[index])
    return evaluate(model, test, device)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="线 A v1：脑区特化 vs 均匀")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="a1")
    args = parser.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    arms = {
        "uniform": (uniform_regions(), 0.0),
        "specialized": (specialized_regions(), 0.75),
    }
    results: dict[str, dict] = {}
    for name, (regions, intra) in arms.items():
        rows = [run_once(regions, intra, seed, device) for seed in SEEDS]
        results[name] = {
            "seed_accuracy": [round(row["accuracy"], 4) for row in rows],
            "seed_firing_rate": [round(row["firing_rate"], 4) for row in rows],
            "accuracy": statistics.median(row["accuracy"] for row in rows),
            "sparsity": 1.0 - statistics.median(row["firing_rate"] for row in rows),
        }
        print(f"[{name}] {results[name]}", flush=True)

    uniform, specialized = results["uniform"], results["specialized"]
    criteria = {
        "h_a0_both_learn": min(uniform["accuracy"], specialized["accuracy"]) >= H_A0_MIN,
        "h_a1_organization_pays": specialized["accuracy"] - uniform["accuracy"] > H_A1_GAP,
        "h_a2_not_denser": specialized["sparsity"] >= uniform["sparsity"],
        "accuracy_gap": round(specialized["accuracy"] - uniform["accuracy"], 4),
    }
    out = {
        "kind": "line A v1: brain-region specialization vs uniform",
        "n_rec": N_REC,
        "fan_in": FAN_IN,
        "epochs": EPOCHS,
        "seeds": list(SEEDS),
        "arms": results,
        "criteria": criteria,
        "line_a_ready": bool(criteria["h_a0_both_learn"] and criteria["h_a1_organization_pays"]),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"[H_A0] 两臂 ≥{H_A0_MIN} → {'过' if criteria['h_a0_both_learn'] else '不过'} | "
        f"[H_A1] 差 {criteria['accuracy_gap']:+.3f} > {H_A1_GAP} → "
        f"{'成立' if criteria['h_a1_organization_pays'] else '不成立'} | "
        f"[H_A2] 稀疏度 {specialized['sparsity']:.3f} vs {uniform['sparsity']:.3f} → "
        f"{'过' if criteria['h_a2_not_denser'] else '不过'}"
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
