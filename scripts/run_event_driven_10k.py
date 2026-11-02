"""事件驱动 e-prop 的 10k 规模训练冒烟：sMNIST，可限样本数，不设验收。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from research.eprop.event_driven import EventDrivenEProp  # noqa: E402
from research.eprop.tasks import load_smnist  # noqa: E402

OUT = REPO / "sweep_results" / "event_driven_10k" / "summary.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="event-driven 10k 规模训练冒烟")
    parser.add_argument("--n-rec", type=int, default=10000)
    parser.add_argument("--fan-in", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train = load_smnist(args.data_dir, split="train", seed=args.seed)
    test = load_smnist(args.data_dir, split="test", seed=args.seed)
    inputs, targets = train.batch(torch.arange(min(args.samples, len(train))))
    inputs, targets = inputs.to(device), targets.to(device)
    n_in = int(inputs.shape[2])
    model = EventDrivenEProp(
        n_in,
        args.n_rec,
        10,
        fan_in=args.fan_in,
        n_refractory=2,
        generator=torch.Generator().manual_seed(args.seed),
    ).to(device)
    started = time.perf_counter()
    losses = []
    for epoch in range(args.epochs):
        permutation = torch.randperm(inputs.shape[1])
        for start in range(0, inputs.shape[1], args.batch_size):
            index = permutation[start : start + args.batch_size]
            loss = model.update(inputs[:, index], targets[index])
            losses.append(loss)
        print(
            f"epoch {epoch + 1}: {len(losses)} updates, last loss {losses[-1]:.4f}, "
            f"elapsed {time.perf_counter() - started:.1f}s",
            flush=True,
        )
    test_inputs, test_labels = test.batch(torch.arange(args.test_samples))
    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
    accuracy = float((model.predict(test_inputs) == test_labels).float().mean())
    elapsed = time.perf_counter() - started
    summary = {
        "kind": "10k event-driven training smoke; no acceptance criterion",
        "config": vars(args) | {"out": str(args.out)},
        "accuracy": accuracy,
        "updates": len(losses),
        "elapsed_s": elapsed,
        "seconds_per_update": elapsed / max(len(losses), 1),
        "sparse_weights": args.n_rec * args.fan_in,
        "peak_mb": (
            torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
