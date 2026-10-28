"""神经发生 v2：NCM 读出（预注册见 sweep_results/neurogenesis_v2_prereg.md）。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from research.common.provenance import collect
from research.common.seeding import parse_seeds
from research.continual.neurogenesis import GrowableNCMReadout
from research.continual.run_neurogenesis import (
    CLASSES_PER_TASK,
    N_IN,
    N_TASKS,
    _tasks,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "sweep_results" / "neurogenesis_ncm"


def run_one(seed: int, *, grow: bool) -> dict:
    tasks = _tasks(seed)
    model = GrowableNCMReadout(
        N_IN,
        N_TASKS * CLASSES_PER_TASK,
        n0=64,
        growth=16,
        max_ratio=2.0,
        min_accuracy=0.90,
        seed=seed,
    )
    history = []
    retention: list[list[float]] = []
    for index, (train_x, train_y, _, _, classes) in enumerate(tasks):
        train_accuracy = model.fit_task(train_x, train_y, classes=classes)
        if grow:
            decision = model.maybe_grow(train_accuracy)
            if decision.grow:
                train_accuracy = model.fit_task(train_x, train_y, classes=classes)
        else:
            decision = None
        seen = [float(np.mean(model.predict(tasks[t][2]) == tasks[t][3])) for t in range(index + 1)]
        retention.append(seen)
        history.append(
            {
                "task": index,
                "train_accuracy": train_accuracy,
                "task_accuracy": seen[-1],
                "n_features": model.n_features,
                "grew": bool(decision.grow) if decision else False,
            }
        )
    return {
        "seed": seed,
        "grow": grow,
        "history": history,
        "retention": retention,
        "final_task_accuracy": retention[-1][-1],
        "min_old_accuracy": min(retention[-1][:-1]) if len(retention[-1]) > 1 else None,
        "n_features": model.n_features,
        "growth_count": sum(1 for decision in model.decisions if decision.grow),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="神经发生 v2：NCM 读出")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    rows = []
    for seed in parse_seeds(args.seeds):
        for grow in (False, True):
            row = run_one(seed, grow=grow)
            rows.append(row)
            print(
                f"seed {seed} grow={grow}: acc {row['retention'][-1]} "
                f"features {row['n_features']} growth {row['growth_count']}",
                flush=True,
            )
    summary = {
        "experiment": "阶段四神经发生 v2：NCM 读出",
        "rows": rows,
        "repro": collect(
            "阶段四神经发生 v2",
            seeds=args.seeds,
            elapsed_s=time.perf_counter() - started,
            notes=["随机 ReLU 特征 + NCM；增长补维、旧类均值补零；非多模态真实插件"],
            device=None,
        ).render_block(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
