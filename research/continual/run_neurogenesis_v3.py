"""神经发生 v3：回放 / 竞争护栏 / 两者并用（预注册见 sweep_results/neurogenesis_v3_prereg.md）。

四个臂共用同一批任务与同一个控制器，只改读出怎么被喂数据、怎么判定归属：

* `fixed`：不增长、不回放、无护栏（对照）；
* `replay`：增长 + 生长后只用回放样本修**新增维度**的旧类原型；
* `margin`：增长 + 新类距离乘 `1+δ`（δ 用旧任务验证集 ∪ 当前任务训练验证片选）；
* `both`：两者一起。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from research.common.provenance import collect
from research.common.seeding import parse_seeds
from research.continual.neurogenesis_v3 import ReplayNCMReadout
from research.continual.run_neurogenesis import CLASSES_PER_TASK, N_IN, N_TASKS, _tasks

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "sweep_results" / "neurogenesis_v3"

ARMS: dict[str, dict[str, bool]] = {
    "fixed": {"grow": False, "replay": False, "margin": False},
    "replay": {"grow": True, "replay": True, "margin": False},
    "margin": {"grow": True, "replay": False, "margin": True},
    "both": {"grow": True, "replay": True, "margin": True},
}

#: 预注册阈值。
FINAL_MIN = 0.90
DROP_MAX_PP = 2.0
MAX_FEATURES = 2 * 64
BUFFER_PER_CLASS = 20


def _split(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple:
    order = rng.permutation(len(x))
    cut = max(1, int(0.8 * len(x)))
    fit, val = order[:cut], order[cut:]
    return x[fit], y[fit], x[val], y[val]


def run_one(seed: int, arm: str, *, buffer_per_class: int = BUFFER_PER_CLASS) -> dict:
    settings = ARMS[arm]
    tasks = _tasks(seed)
    model = ReplayNCMReadout(
        N_IN,
        N_TASKS * CLASSES_PER_TASK,
        n0=64,
        growth=16,
        max_ratio=2.0,
        min_accuracy=0.90,
        buffer_per_class=buffer_per_class,
        use_replay=settings["replay"],
        use_margin=settings["margin"],
        seed=seed,
    )
    rng = np.random.default_rng(seed + 991)
    history = []
    retention: list[list[float]] = []
    validation: list[tuple[np.ndarray, np.ndarray]] = []
    for index, (train_x, train_y, _, _, classes) in enumerate(tasks):
        fit_x, fit_y, val_x, val_y = _split(train_x, train_y, rng)
        accuracy = model.fit_task(fit_x, fit_y, classes=classes, task=index)
        # `fixed` 臂不增长：控制器一次都不许触发，否则对照臂也会长到 128 维。
        decision = model.maybe_grow(accuracy) if settings["grow"] else None
        if decision is not None and decision.grow:
            accuracy = model.fit_task(fit_x, fit_y, classes=classes, task=index)
        validation.append((val_x, val_y))
        margin = model.choose_margin([*validation, (val_x, val_y)], task=index)
        seen = [
            float(np.mean(model.predict(tasks[t][2], task=index) == tasks[t][3]))
            for t in range(index + 1)
        ]
        retention.append(seen)
        history.append(
            {
                "task": index,
                "train_accuracy": accuracy,
                "task_accuracy": seen[-1],
                "n_features": model.n_features,
                "grew": bool(decision.grow) if decision else False,
                "margin": margin,
            }
        )
    buffer_size = sum(len(samples) for samples in model.buffer.values())
    return {
        "seed": seed,
        "arm": arm,
        "history": history,
        "retention": retention,
        "final_task_accuracy": retention[-1][-1],
        "min_old_accuracy": min(retention[-1][:-1]) if len(retention[-1]) > 1 else None,
        "n_features": model.n_features,
        "growth_count": sum(1 for entry in model.stats.decisions if entry.grow),
        "chosen_margins": model.stats.chosen_margins,
        "buffer_size": buffer_size,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="神经发生 v3：回放与竞争护栏")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--arms", default="fixed,replay,margin,both")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    started = time.perf_counter()

    rows = []
    for seed in parse_seeds(args.seeds):
        for arm in args.arms.split(","):
            row = run_one(seed, arm.strip())
            rows.append(row)
            print(
                f"seed {row['seed']} {row['arm']:<7} final {row['final_task_accuracy']:.3f} "
                f"min_old {row['min_old_accuracy'] if row['min_old_accuracy'] is None else round(row['min_old_accuracy'], 3)} "
                f"features {row['n_features']} margins {row['chosen_margins']}",
                flush=True,
            )

    fixed = {(row["seed"]): row for row in rows if row["arm"] == "fixed"}
    per_arm: dict[str, dict] = {}
    for arm in sorted({row["arm"] for row in rows}):
        arm_rows = [row for row in rows if row["arm"] == arm]
        finals = [row["final_task_accuracy"] for row in arm_rows]
        # 逐**旧任务**比对（不是只比最差那个）：判据写的是「所有旧任务下降 ≤2pp」。
        old_drops = [
            (base - value) * 100.0
            for row in arm_rows
            for base, value in zip(
                fixed[row["seed"]]["retention"][-1][:-1], row["retention"][-1][:-1], strict=True
            )
        ]
        criteria = {
            "final_min": float(np.mean(finals)),
            "final_pass": sum(1 for value in finals if value >= FINAL_MIN),
            "drop_max_pp": max(old_drops) if old_drops else None,
            "drop_pass": sum(1 for value in old_drops if value <= DROP_MAX_PP),
            "features_max": max(row["n_features"] for row in arm_rows),
            "features_pass": all(row["n_features"] <= MAX_FEATURES for row in arm_rows),
            "buffer_max": max(row["buffer_size"] for row in arm_rows),
        }
        ready = (
            criteria["final_pass"] == len(arm_rows)
            and criteria["drop_pass"] == len(old_drops)
            and criteria["features_pass"]
        )
        per_arm[arm] = {**criteria, "ready": ready}
    ready_arms = [arm for arm, entry in per_arm.items() if entry["ready"] and arm != "fixed"]

    summary = {
        "kind": "neurogenesis v3: replay + competition guard",
        "n_in": N_IN,
        "n_tasks": N_TASKS,
        "classes_per_task": CLASSES_PER_TASK,
        "buffer_per_class": BUFFER_PER_CLASS,
        "criteria_thresholds": {
            "final_min": FINAL_MIN,
            "drop_max_pp": DROP_MAX_PP,
            "max_features": MAX_FEATURES,
        },
        "arms": per_arm,
        "ready_arms": ready_arms,
        "neurogenesis_v3_ready": bool(ready_arms),
        "rows": rows,
        "repro": collect(
            "阶段四神经发生 v3：回放与竞争护栏",
            seeds=args.seeds,
            notes=[
                "固定 64 → 最大 128 特征；4 任务 × 4 类；NCM 读出",
                "臂：fixed / replay / margin / both；δ 用旧任务验证集 ∪ 当前任务验证片选",
            ],
            device=None,
        ).render_block(),
        "elapsed_s": time.perf_counter() - started,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"arms": per_arm, "ready": ready_arms}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
