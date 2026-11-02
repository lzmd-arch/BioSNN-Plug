"""S4：显存记账（判据见 `research/scale/prereg-s4.md`）。

只跑**一次前向 + 一次 update**，不做训练。变体：n_rec × batch。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research.eprop import EventDrivenEProp, load_smnist

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "sweep_results" / "scale_s4"


def measure(n_rec: int, batch: int, fan_in: int, data_dir: str) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    train = load_smnist(data_dir, split="train", seed=0)
    inputs, targets = train.batch(torch.arange(max(batch, 8)))
    inputs = inputs[:, :batch].cuda()
    targets = targets[:batch].cuda()
    model = EventDrivenEProp(
        int(inputs.shape[2]),
        n_rec,
        10,
        fan_in=fan_in,
        n_refractory=2,
        generator=torch.Generator().manual_seed(0),
    ).cuda()
    before = torch.cuda.memory_allocated() / 2**20
    loss = model.update(inputs, targets)
    peak = torch.cuda.max_memory_allocated() / 2**20
    return {
        "n_rec": n_rec,
        "batch": batch,
        "fan_in": fan_in,
        "t_steps": int(inputs.shape[0]),
        "allocated_before_mb": round(before, 1),
        "peak_mb": round(peak, 1),
        "loss": round(float(loss), 4),
        "weights_mb": round(n_rec * fan_in * 4 / 2**20, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S4 显存记账")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args(argv)

    rows = []
    for n_rec in (8192, 16384):
        for batch in (8, 1):
            row = measure(n_rec, batch, 32, args.data_dir)
            rows.append(row)
            print(row, flush=True)

    # 最大那档的账本细节
    torch.cuda.reset_peak_memory_stats()
    print("=== memory_summary（最后一次配置之后）===")
    print(torch.cuda.memory_summary())

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "accounting.json"
    path.write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
