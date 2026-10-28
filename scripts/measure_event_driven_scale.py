"""事件驱动 e-prop 的规模冒烟：只测**推理**内存/时间，不训练、不报精度。

用途：验证固定 fan-in 的稀疏表示能否在 8 GB 上跑 10k/100k/500k 的前向。训练规模仍受
e-prop 的更新路径限制，本脚本不对此作结论。
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

OUT = REPO / "sweep_results" / "event_driven_scale" / "summary.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="事件驱动 e-prop 推理规模冒烟")
    parser.add_argument("--sizes", default="10000,100000,500000")
    parser.add_argument("--fan-in", type=int, default=32)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--n-in", type=int, default=28)
    parser.add_argument("--n-out", type=int, default=10)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    sizes = [int(item) for item in args.sizes.split(",") if item.strip()]
    results = []
    for n_rec in sizes:
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
        predicted = model.predict(inputs)
        elapsed = time.perf_counter() - started
        results.append(
            {
                "n_rec": n_rec,
                "fan_in": args.fan_in,
                "steps": args.steps,
                "elapsed_s": elapsed,
                "sparse_weights": n_rec * args.fan_in,
                "prediction": int(predicted.item()),
            }
        )
        print(f"n_rec={n_rec}: {elapsed:.2f} s, sparse weights {n_rec * args.fan_in:,}", flush=True)
    summary = {
        "kind": "inference-only scale smoke; no training/accuracy claim",
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
