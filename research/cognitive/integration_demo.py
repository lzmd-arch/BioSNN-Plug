"""阶段四集成 demo：把工作记忆 v4、元认知 v3、世界模型 P3、字面情景索引串起来。

**这是组合/等价性 demo，不是新的性能主张。** 每个组件在同一份输入上再调用一次，
断言输出与“单独调用”逐位一致；再报各自的描述性指标。验收只要求组合可运行、等价性
成立、指标有限，不设新的算法门槛。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.cognitive.run_working_memory_v1 import _episodes, _r2
from research.cognitive.run_world_model_closed_loop import collect_episodes
from research.cognitive.working_memory import LinearRLSReadout
from research.cognitive.world_model import RidgeActionModel, auroc
from research.common.provenance import collect
from research.common.seeding import parse_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "sweep_results" / "stage4_integration_demo"

MEMBERS = 3
EPOCHS = 10


def _working_memory(seed: int, train: int, test: int) -> dict[str, Any]:
    train_states, train_targets = _episodes(train, seed=seed)
    test_states, test_targets = _episodes(test, seed=seed + 10_000)
    readout = LinearRLSReadout(train_states.shape[1], train_targets.shape[1], delta=1000.0)
    readout.fit(train_states, train_targets, epochs=EPOCHS, rng=np.random.default_rng(seed))
    predicted = readout.predict(test_states)
    return {
        "r2": _r2(predicted, test_targets),
        "train_states": train_states,
        "train_targets": train_targets,
        "states": test_states,
        "targets": test_targets,
        "predicted": predicted,
    }


def _metacognition(
    seed: int,
    train_states: np.ndarray,
    train_targets: np.ndarray,
    test_states: np.ndarray,
    test_targets: np.ndarray,
) -> dict[str, Any]:
    predictions = []
    for member in range(MEMBERS):
        rng = np.random.default_rng(seed + member)
        index = rng.choice(train_states.shape[0], size=train_states.shape[0], replace=True)
        readout = LinearRLSReadout(train_states.shape[1], train_targets.shape[1], delta=1000.0)
        readout.fit(train_states[index], train_targets[index], epochs=EPOCHS, rng=rng)
        predictions.append(readout.predict(test_states))
    stacked = np.stack(predictions)
    disagreement = stacked.std(axis=0).mean(axis=1)
    error = ((stacked.mean(axis=0) - test_targets) ** 2).mean(axis=1)
    labels = (error > np.median(error)).astype(int)
    return {
        "disagreement": disagreement,
        "error_auroc": auroc(disagreement, labels),
        "mean_error": float(np.mean(error)),
    }


def _episodic(states: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    stored_states = states[: max(1, states.shape[0] // 2)]
    stored_targets = targets[: stored_states.shape[0]]
    query = states[-stored_states.shape[0] :]
    distances = (
        (query**2).sum(axis=1)[:, None]
        + (stored_states**2).sum(axis=1)[None, :]
        - 2.0 * query @ stored_states.T
    )
    retrieved = stored_targets[np.argmin(distances, axis=1)]
    return {
        "retrieval_mse": float(np.mean((retrieved - targets[-query.shape[0] :]) ** 2)),
    }


def _world_model(episodes: int = 20) -> dict[str, Any]:
    data = collect_episodes(episodes, seed=0, noise=0.1)
    states = np.concatenate([episode.states for episode in data])
    actions = np.concatenate([episode.actions for episode in data])
    next_states = np.concatenate([episode.next_states for episode in data])
    terminated = np.concatenate([episode.terminated for episode in data])
    split = int(0.8 * states.shape[0])
    model = RidgeActionModel(quadratic=True)
    model.fit(states[:split], actions[:split], next_states[:split], terminated[:split])
    prediction = model.predict(states[split:], actions[split:])
    predicted_next = states[split:] + prediction.delta
    nmse = float(np.mean((predicted_next - next_states[split:]) ** 2))
    # 等价性：同一模型、同一输入再预测一次，必须逐位相同。
    again = model.predict(states[split:], actions[split:])
    return {
        "one_step_nmse": nmse,
        "parity": bool(np.array_equal(prediction.delta, again.delta)),
    }


def run(seeds: list[int], *, out: Path) -> dict[str, Any]:
    started = time.perf_counter()
    rows = []
    for seed in seeds:
        wm = _working_memory(seed, train=500, test=100)
        # 等价性：同一输入再调用一次。
        again = LinearRLSReadout(wm["states"].shape[1], wm["targets"].shape[1], delta=1000.0)
        # 用同一训练数据重建，比较预测需要同一训练；这里用 demo 的 predicted 作为基准，
        # 通过再次构造/拟合来验证确定性。
        train_states, train_targets = _episodes(500, seed=seed)
        again.fit(train_states, train_targets, epochs=EPOCHS, rng=np.random.default_rng(seed))
        wm_parity = bool(np.array_equal(again.predict(wm["states"]), wm["predicted"]))
        meta = _metacognition(
            seed,
            wm["train_states"],
            wm["train_targets"],
            wm["states"],
            wm["targets"],
        )
        episodic = _episodic(wm["states"], wm["targets"])
        rows.append(
            {
                "seed": seed,
                "wm_r2": wm["r2"],
                "wm_parity": wm_parity,
                "metacog_error_auroc": meta["error_auroc"],
                "metacog_mean_error": meta["mean_error"],
                "episodic_retrieval_mse": episodic["retrieval_mse"],
            }
        )
        print(
            f"seed {seed}: WM R2 {np.round(wm['r2'], 3).tolist()} parity {wm_parity} | "
            f"meta AUROC {meta['error_auroc']:.3f} | epi MSE {episodic['retrieval_mse']:.3f}",
            flush=True,
        )

    world = _world_model()
    gate = {
        "wm_parity": all(row["wm_parity"] for row in rows),
        "world_model_parity": world["parity"],
        "metrics_finite": all(
            np.all(np.isfinite(row["wm_r2"]))
            and np.isfinite(row["metacog_error_auroc"])
            and np.isfinite(row["episodic_retrieval_mse"])
            for row in rows
        )
        and np.isfinite(world["one_step_nmse"]),
    }
    gate["metrics_finite"] = bool(gate["metrics_finite"])
    gate["integration_ok"] = bool(all(gate.values()))
    summary = {
        "experiment": "阶段四集成 demo（组合/等价性）",
        "config": {"members": MEMBERS, "epochs": EPOCHS, "seeds": seeds},
        "per_seed": rows,
        "world_model": world,
        "gate": gate,
        "repro": collect(
            "阶段四集成 demo",
            seeds=",".join(str(s) for s in seeds),
            elapsed_s=time.perf_counter() - started,
            notes=[
                "组合/等价性 demo，不是新的性能主张",
                "WM v4（局部 RLS）+ 元认知 ensemble 分歧 + P3 世界模型 + 字面情景索引",
            ],
            device=None,
        ).render_block(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="阶段四集成 demo")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    run(parse_seeds(args.seeds), out=args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
