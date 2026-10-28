"""事件驱动 e-prop 的训练规模冒烟：单 batch 一次 update，不报精度。

只验证 10k/100k/500k 的 update 路径能跑，以及时间量级；**不是**一个 epoch 的训练实验。
"""

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

OUT = REPO / "sweep_results" / "event_driven_train_scale" / "summary.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="事件驱动 e-prop 单 batch 训练冒烟")
    parser.add_argument("--sizes", default="10000,100000,500000")
    parser.add_argument("--fan-in", type=int, default=32)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--n-in", type=int, default=28)
    parser.add_argument("--n-out", type=int, default=10)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    results = []
    for n_rec in [int(item) for item in args.sizes.split(",") if item.strip()]:
        started = time.perf_counter()
        model = EventDrivenEProp(
            args.n_in,
            n_rec,
            args.n_out,
            fan_in=args.fan_in,
            n_refractory=2,
            generator=torch.Generator().manual_seed(0),
        )
        inputs = (
            torch.rand(args.steps, 1, args.n_in, generator=torch.Generator().manual_seed(1)) < 0.25
        ).float()
        loss = model.update(inputs, torch.tensor([0]))
        elapsed = time.perf_counter() - started
        results.append(
            {
                "n_rec": n_rec,
                "fan_in": args.fan_in,
                "steps": args.steps,
                "batch": 1,
                "elapsed_s": elapsed,
                "loss": loss,
                "sparse_weights": n_rec * args.fan_in,
            }
        )
        print(f"n_rec={n_rec}: update {elapsed:.2f} s, loss {loss:.4f}", flush=True)
    summary = {
        "kind": "one-update training smoke; NOT an epoch accuracy experiment",
        "config": vars(args) | {"out": str(args.out)},
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
